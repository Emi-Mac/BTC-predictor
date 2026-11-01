"use client";

/** Volatilidad esperada (anual, semanal y diaria), régimen y referencias (DVOL, realizada). */
import { Info } from "lucide-react";
import { useVolatility } from "@/hooks/useVolatility";
import { describeFallback, formatPct, formatVol } from "@/lib/volatility";
import { formatDateTime } from "@/lib/utils";
import { Card } from "@/components/ui/Card";
import { CardSkeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";
import { RegimeBadge } from "./RegimeBadge";
import { ValidationBadge } from "./ValidationBadge";

function Row({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="flex items-center justify-between text-sm" title={hint}>
      <span className="text-gray-400">{label}</span>
      <span className="font-mono font-semibold text-gray-100">{value}</span>
    </div>
  );
}

export function VolatilityCard() {
  const week = useVolatility("7d");
  const day = useVolatility("1d");
  const forecast = week.forecast;

  if (week.isLoading) {
    return (
      <Card title="Volatilidad esperada">
        <CardSkeleton lines={4} />
      </Card>
    );
  }
  if (week.error || !forecast) {
    return (
      <Card title="Volatilidad esperada">
        <ErrorState message="No se pudo cargar el pronóstico de volatilidad." />
      </Card>
    );
  }

  return (
    <Card
      title="Volatilidad esperada"
      action={<ValidationBadge status={forecast.validation_status} />}
    >
      <div className="flex items-end justify-between gap-2">
        <div>
          <p className="font-mono text-3xl font-bold text-gray-100">
            {formatVol(forecast.vol_annual)}
          </p>
          <p className="text-xs text-gray-500">
            anualizada, pronóstico a 7 días
          </p>
        </div>
        <RegimeBadge regime={forecast.regime} />
      </div>

      <div className="flex flex-col gap-1.5">
        <Row
          label="Semanal (±1σ)"
          value={formatPct(forecast.vol_horizon_pct)}
          hint="Desviación estándar esperada del retorno de BTC a 7 días."
        />
        <Row
          label="Diaria (±1σ)"
          value={day.forecast ? formatPct(day.forecast.vol_horizon_pct) : "—"}
          hint="Desviación estándar esperada del retorno de BTC a 1 día."
        />
        <Row
          label="DVOL (implícita, 30d)"
          value={formatVol(forecast.dvol)}
          hint="Volatilidad implícita de opciones (Deribit); referencia, no insumo de la banda."
        />
        <Row label="Realizada 7d" value={formatVol(forecast.realized_vol_7d)} />
        <Row
          label="Realizada 30d"
          value={formatVol(forecast.realized_vol_30d)}
        />
      </div>

      {forecast.fallback && (
        <div className="flex gap-2 rounded-md border border-amber-400/40 bg-amber-400/10 px-3 py-1.5 text-xs text-amber-300">
          <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            Pronóstico de respaldo sin variables exógenas:{" "}
            {describeFallback(forecast.fallback)}.
          </span>
        </div>
      )}

      <p className="text-[11px] text-gray-500">
        Modelo v{forecast.model_version} · vela del{" "}
        {formatDateTime(forecast.candle_open_time)}. El régimen describe el
        nivel de riesgo; no indica hacia dónde se moverá el precio.
      </p>
    </Card>
  );
}
