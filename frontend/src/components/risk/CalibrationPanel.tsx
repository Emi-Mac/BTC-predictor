"use client";

/** Cobertura real de las bandas frente a su nivel nominal, y calidad del pronóstico. */
import { useState } from "react";
import { AlertTriangle, Check, Minus } from "lucide-react";
import type { Horizon } from "@/lib/api";
import { useVolatilityMetrics } from "@/hooks/useVolatility";
import {
  COVERAGE_ALERT_PP,
  HORIZON_LABEL,
  MIN_RESOLVED_FOR_ALERT,
  coverageStatus,
  type CoverageStatus,
} from "@/lib/volatility";
import { formatAccuracy, cn } from "@/lib/utils";
import { Card } from "@/components/ui/Card";
import { CardSkeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";
import { ValidationBadge } from "./ValidationBadge";

const LEVELS = [
  { key: "68", nominal: 0.68 },
  { key: "90", nominal: 0.9 },
  { key: "95", nominal: 0.95 },
] as const;

function StatusIcon({ status }: { status: CoverageStatus }) {
  if (status === "ok")
    return (
      <Check
        className="h-3.5 w-3.5 text-sky-300"
        aria-label="dentro de tolerancia"
      />
    );
  if (status === "alert")
    return (
      <AlertTriangle
        className="h-3.5 w-3.5 text-amber-300"
        aria-label="desviada"
      />
    );
  return (
    <Minus
      className="h-3.5 w-3.5 text-gray-500"
      aria-label="muestra insuficiente"
    />
  );
}

export function CalibrationPanel() {
  const [horizon, setHorizon] = useState<Horizon>("7d");
  const { metrics, error, isLoading } = useVolatilityMetrics(horizon);

  const toggle = (
    <div
      role="group"
      aria-label="Horizonte"
      className="flex rounded-md bg-bg-elevated p-0.5 text-[11px]"
    >
      {(["1d", "7d"] as Horizon[]).map((h) => (
        <button
          key={h}
          type="button"
          aria-pressed={horizon === h}
          onClick={() => setHorizon(h)}
          className={cn(
            "rounded px-2 py-0.5",
            horizon === h
              ? "bg-brand/30 text-gray-100"
              : "text-gray-400 hover:text-gray-200",
          )}
        >
          {h}
        </button>
      ))}
    </div>
  );

  if (isLoading) {
    return (
      <Card title="Calibración" action={toggle}>
        <CardSkeleton lines={4} />
      </Card>
    );
  }
  if (error || !metrics) {
    return (
      <Card title="Calibración" action={toggle}>
        <ErrorState message="No se pudieron cargar las métricas de calibración." />
      </Card>
    );
  }

  const rows = LEVELS.map(({ key, nominal }) => {
    const overall = metrics[`coverage_${key}`];
    const recent = metrics.last_12w[`coverage_${key}`];
    return {
      key,
      nominal,
      overall,
      recent,
      overallStatus: coverageStatus(overall, nominal, metrics.resolved),
      recentStatus: coverageStatus(recent, nominal, metrics.last_12w.resolved),
    };
  });
  const drifting = rows.some(
    (r) => r.overallStatus === "alert" || r.recentStatus === "alert",
  );
  const improvement =
    metrics.qlike !== null &&
    metrics.qlike_naive !== null &&
    metrics.qlike_naive > 0
      ? 1 - metrics.qlike / metrics.qlike_naive
      : null;

  return (
    <Card title="Calibración" action={toggle}>
      <div className="flex items-center justify-between text-xs text-gray-400">
        <span>
          {metrics.resolved} pronósticos resueltos a {HORIZON_LABEL[horizon]}
        </span>
        <ValidationBadge status={metrics.validation_status} />
      </div>

      <table className="w-full text-left text-xs">
        <thead>
          <tr className="text-gray-500">
            <th className="pb-1 font-normal">Banda</th>
            <th className="pb-1 text-right font-normal">Objetivo</th>
            <th className="pb-1 text-right font-normal">Real</th>
            <th className="pb-1 text-right font-normal">12 sem.</th>
          </tr>
        </thead>
        <tbody className="font-mono text-gray-200">
          {rows.map((row) => (
            <tr key={row.key} className="border-t border-border-subtle">
              <td className="py-1.5 text-gray-400">{row.key}%</td>
              <td className="py-1.5 text-right">
                {formatAccuracy(row.nominal)}
              </td>
              <td className="py-1.5 text-right">
                <span className="inline-flex items-center justify-end gap-1">
                  {formatAccuracy(row.overall)}{" "}
                  <StatusIcon status={row.overallStatus} />
                </span>
              </td>
              <td className="py-1.5 text-right">
                <span className="inline-flex items-center justify-end gap-1">
                  {formatAccuracy(row.recent)}{" "}
                  <StatusIcon status={row.recentStatus} />
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {drifting && (
        <div className="rounded-md border border-amber-400/40 bg-amber-400/10 px-3 py-1.5 text-xs text-amber-300">
          La cobertura real se desvía más de {COVERAGE_ALERT_PP} pp del nivel
          nominal: las bandas podrían estar descalibradas. Tómalas con cautela.
        </div>
      )}

      <div className="flex flex-col gap-1 text-xs text-gray-400">
        <div className="flex justify-between">
          <span>QLIKE (modelo / ingenuo)</span>
          <span className="font-mono text-gray-200">
            {metrics.qlike?.toFixed(3) ?? "—"} /{" "}
            {metrics.qlike_naive?.toFixed(3) ?? "—"}
            {improvement !== null &&
              ` (${improvement >= 0 ? "−" : "+"}${Math.abs(improvement * 100).toFixed(0)}%)`}
          </span>
        </div>
        <div
          className="flex justify-between"
          title="Ideal ≈ 1; requiere una muestra amplia."
        >
          <span>Pendiente Mincer-Zarnowitz</span>
          <span className="font-mono text-gray-200">
            {metrics.mz_slope?.toFixed(2) ?? "—"}
          </span>
        </div>
      </div>

      <p className="text-[11px] text-gray-500">
        «—» indica muestra insuficiente (menos de {MIN_RESOLVED_FOR_ALERT}{" "}
        pronósticos). Los pronósticos consecutivos se solapan, por lo que la
        muestra efectiva es menor que el conteo.
      </p>
    </Card>
  );
}
