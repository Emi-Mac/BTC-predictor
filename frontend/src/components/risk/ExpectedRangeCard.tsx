"use client";

/** Rango de precio probable a 1 y 7 días (68/90/95%), con multiplicadores calibrados. */
import type { Horizon, PriceBands } from "@/lib/api";
import { useVolatility } from "@/hooks/useVolatility";
import { HORIZON_LABEL, OUTSIDE_95_NOTE } from "@/lib/volatility";
import { formatCurrency } from "@/lib/utils";
import { Card } from "@/components/ui/Card";
import { CardSkeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";
import { ValidationBadge } from "./ValidationBadge";

const LEVELS: { key: keyof PriceBands; label: string }[] = [
  { key: "p68", label: "68%" },
  { key: "p90", label: "90%" },
  { key: "p95", label: "95%" },
];
const HORIZONS: Horizon[] = ["1d", "7d"];

export function ExpectedRangeCard() {
  const day = useVolatility("1d");
  const week = useVolatility("7d");
  const reference = week.forecast ?? day.forecast;

  if (day.isLoading || week.isLoading) {
    return (
      <Card title="Rango de precio probable">
        <CardSkeleton lines={4} />
      </Card>
    );
  }
  if (!reference) {
    return (
      <Card title="Rango de precio probable">
        <ErrorState message="No se pudo cargar el rango esperado." />
      </Card>
    );
  }

  const forecasts = { "1d": day.forecast, "7d": week.forecast };
  const calibrated = HORIZONS.some((h) => forecasts[h]?.bands);

  return (
    <Card
      title="Rango de precio probable"
      action={<ValidationBadge status={reference.validation_status} />}
    >
      <p className="text-xs text-gray-500">
        Desde {formatCurrency(reference.reference_price, 0)} (cierre de la
        última vela). Multiplicadores calibrados con resultados reales, no
        gaussianos.
      </p>

      {calibrated ? (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="text-gray-500">
                <th className="pb-1 font-normal">Nivel</th>
                {HORIZONS.map((h) => (
                  <th key={h} className="pb-1 text-right font-normal">
                    {HORIZON_LABEL[h]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="font-mono text-gray-200">
              {LEVELS.map(({ key, label }) => (
                <tr key={key} className="border-t border-border-subtle">
                  <td className="py-1.5 text-gray-400">{label}</td>
                  {HORIZONS.map((h) => {
                    const band = forecasts[h]?.bands?.[key];
                    return (
                      <td key={h} className="py-1.5 text-right">
                        {band
                          ? `${formatCurrency(band.low, 0)} – ${formatCurrency(band.high, 0)}`
                          : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="rounded-md border border-border bg-bg-elevated px-3 py-2 text-xs text-gray-400">
          Aún no hay calibración suficiente para mostrar rangos fiables.
        </p>
      )}

      <p className="text-[11px] text-gray-500">
        {OUTSIDE_95_NOTE} No es una garantía ni una señal.
      </p>
    </Card>
  );
}
