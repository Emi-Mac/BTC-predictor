"use client";

/** Panel de métricas: accuracy overall/7d/30d, badge del modelo y sparkline mensual. */
import useSWR from "swr";
import { AreaChart, Area, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import { endpoints, fetcher, type AccuracyMetrics, type ModelMetadata } from "@/lib/api";
import { formatAccuracy } from "@/lib/utils";
import { useEffect } from "react";
import { useMetricsStore } from "@/store/useMetricsStore";
import { Card } from "@/components/ui/Card";
import { CardSkeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";

function MetricRow({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-gray-400">{label}</span>
      <span className="font-mono font-semibold text-gray-100">{formatAccuracy(value)}</span>
    </div>
  );
}

export function AccuracyPanel() {
  const { data: accuracy, error, isLoading } = useSWR<AccuracyMetrics>(
    endpoints.metricsAccuracy,
    fetcher,
    { refreshInterval: 5 * 60_000 },
  );
  const { data: model } = useSWR<ModelMetadata>(endpoints.metricsModel, fetcher, {
    refreshInterval: 5 * 60_000,
  });
  const setDriftAlert = useMetricsStore((s) => s.setDriftAlert);
  const driftAlert = useMetricsStore((s) => s.driftAlert);

  useEffect(() => {
    if (accuracy?.last_30d != null && model?.baseline_accuracy != null) {
      setDriftAlert(accuracy.last_30d < model.baseline_accuracy);
    }
  }, [accuracy?.last_30d, model?.baseline_accuracy, setDriftAlert]);

  if (isLoading) {
    return (
      <Card title="Métricas">
        <CardSkeleton lines={4} />
      </Card>
    );
  }
  if (error || !accuracy) {
    return (
      <Card title="Métricas">
        <ErrorState message="No se pudieron cargar las métricas." />
      </Card>
    );
  }

  const sparklineData = accuracy.by_month.map((m) => ({
    month: m.month,
    accuracy: m.accuracy != null ? (m.accuracy <= 1 ? m.accuracy * 100 : m.accuracy) : null,
  }));

  return (
    <Card
      title="Métricas"
      action={
        model && (
          <span className="rounded-full bg-bg-elevated px-2 py-0.5 text-[11px] text-gray-300">
            Modelo v{model.version}
          </span>
        )
      }
    >
      <div className="flex flex-col gap-1.5">
        <MetricRow label="Últimos 7 días" value={accuracy.last_7d} />
        <MetricRow label="Últimos 30 días" value={accuracy.last_30d} />
        <MetricRow label="Histórico total" value={accuracy.overall} />
        {accuracy.baseline_overall != null && (
          <MetricRow label="Regla trivial (misma muestra)" value={accuracy.baseline_overall} />
        )}
        <p className="text-right text-[11px] text-gray-500">
          {accuracy.total_resolved} predicciones resueltas
        </p>
      </div>

      {driftAlert && (
        <div className="rounded-md border border-neutral/40 bg-neutral-dim/30 px-3 py-1.5 text-xs text-neutral">
          ⚠️ Accuracy 30d por debajo del baseline del modelo — posible model drift.
        </div>
      )}

      {sparklineData.length > 1 && (
        <div className="h-16 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={sparklineData} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#818cf8" stopOpacity={0.5} />
                  <stop offset="100%" stopColor="#818cf8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <YAxis domain={[40, 70]} hide />
              <Tooltip
                contentStyle={{
                  background: "#111827",
                  border: "1px solid #374151",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(v: number) => [`${v.toFixed(1)}%`, "Accuracy"]}
                labelFormatter={(label) => label}
              />
              <Area
                type="monotone"
                dataKey="accuracy"
                stroke="#818cf8"
                fill="url(#accuracyGradient)"
                strokeWidth={2}
                connectNulls
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  );
}



