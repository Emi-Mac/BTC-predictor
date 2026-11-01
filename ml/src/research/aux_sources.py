"""Captura inmutable de fuentes auxiliares para la Fase 2, separada del dataset congelado.

Fuentes:
- Binance (data.binance.vision): velas 4h spot BTCUSDT (historia previa para ventanas largas),
  perpetuo BTCUSDT, índice de prima del perpetuo y spot ETHBTC. Archivos mensuales con checksum.
- CoinMetrics Community API: métricas on-chain diarias de BTC y oferta de USDT/USDC.
- Reserva Federal: calendario de reuniones programadas del FOMC (publicado con antelación).

Las fuentes se guardan crudas con su huella; los Parquet derivados y el manifiesto permiten
verificar que un experimento usó exactamente los mismos datos.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.ingestion.run import configure_logging
from src.research.source_snapshot import file_sha256

log = structlog.get_logger(__name__)
BINANCE_BASE = "https://data.binance.vision/data"
KLINE_SERIES: dict[str, tuple[str, str]] = {
    "btc_spot_klines": ("spot/monthly/klines", "BTCUSDT"),
    "perp_klines": ("futures/um/monthly/klines", "BTCUSDT"),
    "premium_index": ("futures/um/monthly/premiumIndexKlines", "BTCUSDT"),
    "ethbtc_klines": ("spot/monthly/klines", "ETHBTC"),
}
KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
BTC_METRICS = (
    "AdrActCnt",
    "CapMVRVCur",
    "FeeTotNtv",
    "FlowInExUSD",
    "FlowOutExUSD",
    "HashRate",
    "SplyExNtv",
    "TxCnt",
)
STABLECOINS = ("usdt", "usdc")
FOMC_URLS = {
    "fomc_current": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "fomc_2020": "https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm",
}
USER_AGENT = "Mozilla/5.0 (btc-predictor research snapshot)"
FOMC_STATEMENT_TIME = time(14, 0)
NEW_YORK = ZoneInfo("America/New_York")
MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}
EXCLUDED_MEETINGS = ("unscheduled", "notation", "cancelled", "conference call")
AVAILABILITY = {
    "binance_klines": "precio final al cierre de la vela; sin revisiones",
    "premium_index": "valor final al cierre de la vela; sin revisiones",
    "coinmetrics": (
        "valor del día D disponible desde D+2 00:00 UTC (supuesto conservador); "
        "revisiones históricas desconocidas"
    ),
    "fomc": "fechas programadas publicadas con antelación; decisión a las 14:00 de Nueva York",
}


# ─── Parsers puros ───────────────────────────────────────────────────────────


def month_range(start: date, end_exclusive: date) -> list[str]:
    if end_exclusive <= start:
        raise ValueError("El rango de meses está vacío")
    months = pd.period_range(start, end_exclusive - pd.Timedelta(days=1), freq="M")
    return [str(month) for month in months]


def parse_kline_csv(content: bytes) -> pd.DataFrame:
    """Lee un CSV de velas de Binance con o sin encabezado, en ms o µs."""
    text = content.decode("utf-8")
    first = text.split("\n", 1)[0]
    header = 0 if first.startswith("open_time") else None
    frame = pd.read_csv(io.StringIO(text), header=header)
    frame = frame.iloc[:, :6]
    frame.columns = KLINE_COLUMNS
    raw = frame["open_time"].astype("int64")
    unit = np.where(raw > 10**14, 1_000, 1)
    frame["open_time"] = pd.to_datetime(raw // unit, unit="ms", utc=True)
    for column in KLINE_COLUMNS[1:]:
        frame[column] = frame[column].astype(float)
    return frame


def _zip_payload(content: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("El archivo de Binance debe contener un único CSV")
        return archive.read(names[0])


def _month_number(text: str) -> int:
    prefix = text.strip().lower()[:3]
    matches = [number for name, number in MONTHS.items() if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Mes no reconocido: {text}")
    return matches[0]


def _meeting_end(month_text: str, days_text: str, year: int) -> date:
    """Último día de la reunión; admite 'Apr/May' + '30-1' y '30-May 1'."""
    months = [part for part in re.split(r"[/\s\-–]+", month_text) if part.isalpha()]
    months += [
        part for part in re.findall(r"[A-Za-z]+", days_text) if part.lower()[:3] in _abbrev()
    ]
    days = re.findall(r"\d+", days_text)
    if not months or not days:
        raise ValueError(f"Fecha de reunión no reconocida: {month_text} {days_text}")
    return date(year, _month_number(months[-1]), int(days[-1]))


def _abbrev() -> set[str]:
    return {name[:3] for name in MONTHS}


def parse_fomc_calendar(html: str) -> list[date]:
    """Reuniones programadas de la página de calendarios vigente (varios años)."""
    meetings: list[date] = []
    sections = re.split(r'<a id="\d+">(\d{4}) FOMC Meetings', html)
    for index in range(1, len(sections), 2):
        year, body = int(sections[index]), sections[index + 1]
        pattern = (
            r"fomc-meeting__month[^>]*>(?:<strong>)?([^<]+)<.*?"
            r"fomc-meeting__date[^>]*>([^<]+)<"
        )
        for month_text, days_text in re.findall(pattern, body, flags=re.S):
            if any(word in days_text.lower() for word in EXCLUDED_MEETINGS):
                continue
            meetings.append(_meeting_end(month_text, days_text, year))
    return sorted(set(meetings))


def parse_fomc_historical(html: str) -> list[date]:
    """Reuniones programadas de una página histórica anual ('July 28-29 Meeting - 2020')."""
    meetings: list[date] = []
    for heading in re.findall(r"<h5[^>]*>([^<]+)</h5>", html):
        text = heading.strip()
        match = re.match(r"([A-Za-z]+)\s+(.+?)\s+Meeting\s+-\s+(\d{4})$", text)
        if not match or any(word in text.lower() for word in EXCLUDED_MEETINGS):
            continue
        month_text, days_text, year = match.groups()
        meetings.append(_meeting_end(month_text, days_text, int(year)))
    return sorted(set(meetings))


def fomc_decisions(meeting_ends: list[date]) -> pd.DataFrame:
    if not meeting_ends:
        raise ValueError("El calendario del FOMC está vacío")
    times = [
        datetime.combine(day, FOMC_STATEMENT_TIME, tzinfo=NEW_YORK).astimezone(UTC)
        for day in sorted(set(meeting_ends))
    ]
    return pd.DataFrame(
        {
            "meeting_end_date": sorted(set(meeting_ends)),
            "decision_time": pd.to_datetime(times, utc=True),
        }
    )


def parse_coinmetrics(pages: list[dict[str, Any]], metrics: tuple[str, ...]) -> pd.DataFrame:
    rows = [row for page in pages for row in page.get("data", [])]
    if not rows:
        raise ValueError("CoinMetrics no devolvió datos")
    frame = pd.DataFrame(rows)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    for metric in metrics:
        if metric not in frame:
            raise ValueError(f"CoinMetrics no devolvió la métrica {metric}")
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    columns = ["time", *(["asset"] if "asset" in frame else []), *metrics]
    return frame[columns].sort_values(columns[: -len(metrics)]).reset_index(drop=True)


# ─── Descarga ────────────────────────────────────────────────────────────────


class NotFoundError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=20),
    reraise=True,
)
def _get(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> bytes:
    response = client.get(url, params=params)
    if response.status_code == 404:
        raise NotFoundError(url)
    response.raise_for_status()
    return response.content


def _store_raw(directory: Path, name: str, content: bytes) -> str:
    path = directory / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def _capture_klines(
    client: httpx.Client, directory: Path, name: str, months: list[str], raw: dict[str, Any]
) -> pd.DataFrame:
    prefix, symbol = KLINE_SERIES[name]
    frames: list[pd.DataFrame] = []
    missing_leading: list[str] = []
    for month in months:
        filename = f"{symbol}-4h-{month}.zip"
        url = f"{BINANCE_BASE}/{prefix}/{symbol}/4h/{filename}"
        try:
            content = _get(client, url)
        except NotFoundError:
            if frames:
                raise ValueError(f"Falta un mes intermedio de {name}: {month}") from None
            missing_leading.append(month)
            continue
        expected = _get(client, url + ".CHECKSUM").decode("utf-8").split()[0]
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected:
            raise ValueError(f"Checksum inválido para {url}")
        stored = f"{name}/{filename}"
        _store_raw(directory, stored, content)
        raw[stored] = {"url": url, "sha256": digest, "checksum_verified": True}
        frames.append(parse_kline_csv(_zip_payload(content)))
    if not frames:
        raise ValueError(f"No hay datos de {name}")
    frame = pd.concat(frames, ignore_index=True).sort_values("open_time")
    if frame["open_time"].duplicated().any():
        raise ValueError(f"Velas duplicadas en {name}")
    frame.attrs["missing_leading_months"] = missing_leading
    return frame.reset_index(drop=True)


def _capture_coinmetrics(
    client: httpx.Client,
    directory: Path,
    name: str,
    assets: tuple[str, ...],
    metrics: tuple[str, ...],
    start: date,
    end: date,
    raw: dict[str, Any],
) -> pd.DataFrame:
    params: dict[str, Any] | None = {
        "assets": ",".join(assets),
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "start_time": start.isoformat(),
        "end_time": (end - pd.Timedelta(days=1)).isoformat(),
        "page_size": 10000,
    }
    url = COINMETRICS_URL
    pages: list[dict[str, Any]] = []
    while url:
        content = _get(client, url, params)
        index = len(pages)
        stored = f"{name}/page-{index:03d}.json"
        raw[stored] = {
            "url": url if params is None else f"{url}?{httpx.QueryParams(params)}",
            "sha256": _store_raw(directory, stored, content),
            "checksum_verified": False,
        }
        page = json.loads(content)
        pages.append(page)
        url, params = page.get("next_page_url", ""), None
    return parse_coinmetrics(pages, metrics)


def _capture_fomc(client: httpx.Client, directory: Path, raw: dict[str, Any]) -> pd.DataFrame:
    meetings: list[date] = []
    for name, url in FOMC_URLS.items():
        content = _get(client, url)
        stored = f"fomc/{name}.html"
        raw[stored] = {
            "url": url,
            "sha256": _store_raw(directory, stored, content),
            "checksum_verified": False,
        }
        html = content.decode("utf-8", errors="replace")
        parsed = (
            parse_fomc_calendar(html) if name == "fomc_current" else parse_fomc_historical(html)
        )
        if not parsed:
            raise ValueError(f"No se reconocieron reuniones en {url}")
        meetings.extend(parsed)
    return fomc_decisions(meetings)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    content = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def capture(directory: Path, aux_id: str, start: date, end: date) -> dict[str, Any]:
    if (directory / "manifest.json").exists():
        raise ValueError("Las fuentes auxiliares ya fueron finalizadas; usar otro ID")
    directory.mkdir(parents=True, exist_ok=True)
    months = month_range(start, end)
    raw: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    with httpx.Client(timeout=60, headers={"User-Agent": USER_AGENT}) as client:
        for name in KLINE_SERIES:
            frames[name] = _capture_klines(client, directory, name, months, raw)
            log.info("aux_klines_captured", source=name, rows=len(frames[name]))
        frames["coinmetrics_btc"] = _capture_coinmetrics(
            client, directory, "coinmetrics_btc", ("btc",), BTC_METRICS, start, end, raw
        )
        frames["coinmetrics_stablecoins"] = _capture_coinmetrics(
            client,
            directory,
            "coinmetrics_stablecoins",
            STABLECOINS,
            ("SplyCur",),
            start,
            end,
            raw,
        )
        frames["fomc"] = _capture_fomc(client, directory, raw)
    files = {}
    coverage = {}
    for name, frame in frames.items():
        path = directory / f"{name}.parquet"
        temporary = directory / f".{name}.parquet.part"
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        files[name] = {"path": path.name, "sha256": file_sha256(path), "rows": len(frame)}
        column = next(c for c in ("open_time", "time", "decision_time") if c in frame)
        coverage[name] = {
            "first": frame[column].min().isoformat(),
            "last": frame[column].max().isoformat(),
            "missing_leading_months": frame.attrs.get("missing_leading_months", []),
        }
    manifest = {
        "version": 1,
        "aux_id": aux_id,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "captured_at": datetime.now(UTC).isoformat(),
        "point_in_time_verified": False,
        "availability_assumptions": AVAILABILITY,
        "implementation_sha256": file_sha256(Path(__file__)),
        "coverage": coverage,
        "files": files,
        "raw": raw,
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    temporary = directory / ".manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    log.info("aux_sources_captured", aux_id=aux_id, coverage=coverage)
    return manifest


def verify(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError("Falta el manifiesto de fuentes auxiliares")
    manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_sha256")
    if _manifest_digest(manifest) != expected:
        raise ValueError("La huella del manifiesto auxiliar no coincide")
    manifest["manifest_sha256"] = expected
    root = directory.resolve()
    for relative, item in [
        *((item["path"], item) for item in manifest["files"].values()),
        *((f"raw/{name}", item) for name, item in manifest["raw"].items()),
    ]:
        target = (directory / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"Falta el archivo auxiliar {relative}")
        if file_sha256(target) != item["sha256"]:
            raise ValueError(f"La huella de {relative} no coincide")
    return manifest


def load(directory: Path) -> dict[str, pd.DataFrame]:
    manifest = verify(directory)
    return {
        name: pd.read_parquet(directory / item["path"]) for name, item in manifest["files"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuentes auxiliares inmutables para la Fase 2")
    parser.add_argument("command", choices=["capture", "verify"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--aux-id")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    configure_logging(get_settings().log_level)
    if args.command == "capture":
        if not (args.aux_id and args.start and args.end):
            parser.error("capture requiere --aux-id, --start y --end")
        capture(args.directory, args.aux_id, args.start, args.end)
    else:
        manifest = verify(args.directory)
        log.info("aux_sources_verified", aux_id=manifest["aux_id"])


if __name__ == "__main__":
    main()
