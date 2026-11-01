/**
 * Presentación de la volatilidad. La volatilidad no es una señal direccional: no se usan los
 * colores de compra/venta (verde/rojo) y los textos no prometen resultados.
 */
import type { Horizon, VolatilityRegime } from "@/lib/api";

export const HORIZON_LABEL: Record<Horizon, string> = {
  "1d": "1 día",
  "7d": "7 días",
};

export const REGIME_LABEL: Record<VolatilityRegime, string> = {
  low: "Bajo",
  normal: "Normal",
  high: "Alto",
  extreme: "Extremo",
};

/** Clases de insignia por régimen (azul, índigo, ámbar, violeta: ninguna es verde ni roja). */
export const REGIME_CLASS: Record<VolatilityRegime, string> = {
  low: "border-sky-400/50 bg-sky-400/10 text-sky-300",
  normal: "border-brand/50 bg-brand/10 text-brand",
  high: "border-amber-400/50 bg-amber-400/10 text-amber-300",
  extreme: "border-fuchsia-400/50 bg-fuchsia-400/10 text-fuchsia-300",
};

export const FALLBACK_REASON_LABEL: Record<string, string> = {
  dvol_stale: "DVOL desactualizado",
  features_missing: "faltan features de la vela",
  funding_stale: "funding desactualizado",
  insufficient_history: "historial insuficiente",
  fomc_calendar_missing: "calendario FOMC sin reuniones futuras",
};

/** Traduce el `fallback_reason` (códigos separados por comas) a texto legible. */
export function describeFallback(reason: string): string {
  return reason
    .split(",")
    .map((code) => FALLBACK_REASON_LABEL[code.trim()] ?? code.trim())
    .join(", ");
}

/** Fracción (0.45) → "45.0%". */
export function formatVol(
  value: number | null | undefined,
  decimals = 1,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Valor ya en puntos porcentuales (6.9) → "6.9%". */
export function formatPct(
  value: number | null | undefined,
  decimals = 1,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value.toFixed(decimals)}%`;
}

export const COVERAGE_ALERT_PP = 5;
export const MIN_RESOLVED_FOR_ALERT = 30;

export type CoverageStatus = "ok" | "alert" | "insufficient";

/** Cobertura real frente al nivel nominal: alerta si se desvía más de 5 pp con muestra suficiente. */
export function coverageStatus(
  actual: number | null,
  nominal: number,
  resolved: number,
): CoverageStatus {
  if (actual === null || resolved < MIN_RESOLVED_FOR_ALERT)
    return "insufficient";
  return Math.abs(actual - nominal) * 100 > COVERAGE_ALERT_PP ? "alert" : "ok";
}

/** Un cierre fuera del intervalo «del 95%» ocurre 1 de cada 20 veces, no es un error del modelo. */
export const OUTSIDE_95_NOTE =
  "Aun con el rango del 95%, aproximadamente 1 de cada 20 semanas el precio termina fuera de él.";
