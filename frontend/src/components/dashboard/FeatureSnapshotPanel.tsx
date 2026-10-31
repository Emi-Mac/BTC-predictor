"use client";

/**
 * Panel de transparencia de features: el backend expone los valores actuales
 * de las features calculadas (no un ranking de importancia SHAP, que no está
 * disponible en la API). Mostramos el top-10 por magnitud absoluta normalizada
 * como proxy visual de qué señales están más "activas" en este momento.
 */
import useSWR from "swr";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from "recharts";
import { endpoints, fetcher, type FeatureSnapshot } from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { CardSkeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";

export function FeatureSnapshotPanel() {
  const { data, error, isLoading } = useSWR<FeatureSnapshot>(
    endpoints.featuresLatest,
    fetcher,
    { refreshInterval: 5 * 60_000 },
  );

  if (isLoading) {
    return (
      <Card title="Features Actuales">
        <CardSkeleton lines={5} />
      </Card>
    );
  }
  if (error || !data) {
    return (
      <Card title="Features Actuales">
        <ErrorState message="No se pudieron cargar las features." />
      </Card>
    );
  }

  const numericEntries = Object.entries(data.values).filter(
    (entry): entry is [string, number] => typeof entry[1] === "number",
  );
  const top = numericEntries
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 10)
    .map(([name, value]) => ({ name, value }))
    .reverse();

  return (
    <Card title="Features Actuales (Transparencia)">
      {top.length === 0 ? (
        <ErrorState message="No hay features numéricas disponibles." />
      ) : (
        <div style={{ height: Math.max(180, top.length * 28) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={top} layout="vertical" margin={{ left: 8, right: 16 }}>
              <XAxis type="number" hide />
              <YAxis
                type="category"
                dataKey="name"
                width={120}
                tick={{ fill: "#9ca3af", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  background: "#111827",
                  border: "1px solid #374151",
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(v: number) => [v.toFixed(4), "Valor"]}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={14}>
                {top.map((entry, i) => (
                  <Cell key={i} fill={entry.value >= 0 ? "#4ade80" : "#f87171"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <p className="text-[11px] text-gray-500">
        Vela: {new Date(data.open_time).toLocaleString("es-ES")}
      </p>
    </Card>
  );
}

