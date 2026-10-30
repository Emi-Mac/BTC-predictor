import hashlib
import io
import zipfile
from datetime import date

import httpx
import pytest

from src.research.metrics_archive import (
    METRICS_COLUMNS,
    MetricsArchiveClient,
    checksum_from_text,
    parse_daily_archive,
)


def archive_bytes(day="2020-09-01", duplicate=True, conflict=False):
    header = ",".join(["create_time", "symbol", *METRICS_COLUMNS])
    first = f"{day} 00:00:00,BTCUSDT,100,1000,1.1,1.2,1.3,1.4"
    second = f"{day} 00:05:00,BTCUSDT,101,1010,1.1,1.2,1.3,1.4"
    lines = [header, second, first]
    if duplicate:
        lines.append(first if not conflict else first.replace(",100,1000,", ",102,1000,"))
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr(f"BTCUSDT-metrics-{day}.csv", "\n".join(lines))
    return content.getvalue()


def test_daily_archive_deduplicates_only_identical_rows_and_audits_gaps():
    frame, audit = parse_daily_archive(archive_bytes(), "BTCUSDT", date(2020, 9, 1))
    assert audit["raw_rows"] == 3
    assert audit["duplicates_removed"] == 1
    assert audit["unique_rows"] == 2
    assert audit["missing_intervals"] == 286
    assert frame["create_time"].is_monotonic_increasing
    assert str(frame["create_time"].dt.tz) == "UTC"
    with pytest.raises(ValueError, match="contradictorios"):
        parse_daily_archive(archive_bytes(conflict=True), "BTCUSDT", date(2020, 9, 1))


def test_checksum_filename_is_validated():
    digest = "a" * 64
    assert checksum_from_text(f"{digest}  file.zip", "file.zip") == digest
    with pytest.raises(ValueError, match="otro archivo"):
        checksum_from_text(f"{digest}  other.zip", "file.zip")
    with pytest.raises(ValueError, match="inválido"):
        checksum_from_text("not-a-hash", "file.zip")


async def test_archive_download_checks_checksum_and_detects_cache_tampering(tmp_path):
    data = archive_bytes()
    digest = hashlib.sha256(data).hexdigest()
    filename = "BTCUSDT-metrics-2020-09-01.zip"
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=f"{digest}  {filename}")
        return httpx.Response(
            200, content=data, headers={"Last-Modified": "Tue, 01 Sep 2020 00:00:00 GMT"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = MetricsArchiveClient(http, min_interval_seconds=0)
        metadata = await client.fetch_day(tmp_path, "BTCUSDT", date(2020, 9, 1))
        assert metadata["sha256"] == digest
        assert len(calls) == 2
        assert await client.fetch_day(tmp_path, "BTCUSDT", date(2020, 9, 1)) == metadata
        assert len(calls) == 2
        (tmp_path / filename).write_bytes(b"tampered")
        with pytest.raises(ValueError, match="no coincide"):
            await client.fetch_day(tmp_path, "BTCUSDT", date(2020, 9, 1))


async def test_archive_rejects_bad_checksum_without_writing_data(tmp_path):
    data = archive_bytes()
    filename = "BTCUSDT-metrics-2020-09-01.zip"

    def handler(request):
        if request.url.path.endswith(".CHECKSUM"):
            return httpx.Response(200, text=f"{'a' * 64}  {filename}")
        return httpx.Response(200, content=data)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = MetricsArchiveClient(http, min_interval_seconds=0)
        with pytest.raises(ValueError, match="checksum"):
            await client.fetch_day(tmp_path, "BTCUSDT", date(2020, 9, 1))
    assert list(tmp_path.iterdir()) == []
