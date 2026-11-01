"use client";

/** Volatilidad pronosticada a 7 días frente a la realizada, de los pronósticos ya resueltos. */
import { useMemo } from "react";
import useSWR from "swr";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  endpoints,
  fetcher,
  isNotFound,
  type Page,
  type VolatilityHistoryPoint,
} from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";

const HORIZON = "7d" as const;
const POINTS = 500; // máximo por solicitud del backend
const WEEKS = 26;
const WEEK_MS = 7 * 24 * 3600 * 1000;

interface ChartPoint {
  time: number;
  forecast: number;
  realized: number | null;
}

export function VolatilityHistoryChart() {
  const { data, error, isLoading } = useSWR<Page<VolatilityHistoryPoint>>(
    endpoints.volatilityHistory(HORIZON, POINTS, 0),
    fetcher,
    {
      refreshInterval: 5 * 60_000,
      shouldRetryOnError: (e: unknown) => !isNotFound(e),
    },
  );

  const points = useMemo<ChartPoint[]>(() => {
    if (!data) return [];
    const newest = Math.max(
      ...data.data.map((p) => new Date(p.candle_open_time).getTime()),
    );
    return data.data
      .map((p) => ({
        time: new Date(p.candle_open_time).getTime(),
        forecast: p.forecast_vol_annual * 100,
        realized:
          p.realized_vol_annual === null ? null : p.realized_vol_annual * 100,
      }))
      .filter((p) => p.time >= newest - WEEKS * WEEK_MS)
      .sort((a, b) => a.time - b.time);
  }, [data]);

  const resolved = points.filter((p) => p.realized !== null).length;

  return (
    <Card
      title="Volatilidad: pronosticada vs realizada (7d)"
      className="h-full"
    >
      {isLoading ? (
        <Skeleton className="h-64 w-full" />
      ) : error || points.length === 0 ? (
        <ErrorState
          message={
            error
              ? "No se pudo cargar el historial de volatilidad."
              : "Aún no hay historial."
          }
        />
      ) : (
        <>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={points}
                margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
              >
                <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                <XAxis
                  dataKey="time"
                  type="number"
                  scale="time"
                  domain={["dataMin", "dataMax"]}
                  tickFormatter={(t: number) =>
                    new Date(t).toLocaleDateString("es-ES", {
                      day: "2-digit",
                      month: "short",
                    })
                  }
                  stroke="#6b7280"
                  fontSize={11}
                />
                <YAxis
                  stroke="#6b7280"
                  fontSize={11}
                  width={42}
                  tickFormatter={(v: number) => `${v.toFixed(0)}%`}
                  domain={["auto", "auto"]}
                />
                <Tooltip
                  contentStyle={{
                    background: "#111827",
                    border: "1px solid #374151",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  labelFormatter={(t: number) =>
                    new Date(t).toLocaleString("es-ES")
                  }
                  formatter={(v: number, name: string) => [
                    `${v.toFixed(1)}%`,
                    name,
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  name="Pronosticada"
                  type="monotone"
                  dataKey="forecast"
                  stroke="#818cf8"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  name="Realizada"
                  type="monotone"
                  dataKey="realized"
                  stroke="#cbd5e1"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  dot={false}
                  connectNulls={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-gray-500">
            Cada punto compara el pronóstico emitido en esa vela con la
            volatilidad realizada en los 7 días siguientes ({resolved} de{" "}
            {points.length} ya resueltos; el tramo más reciente aún no tiene
            valor realizado). Volatilidad anualizada.
          </p>
        </>
      )}
    </Card>
  );
}
