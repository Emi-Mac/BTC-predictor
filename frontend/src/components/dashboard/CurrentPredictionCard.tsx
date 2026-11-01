"use client";

/** Card prominente con la predicción activa: dirección, confianza y countdown al cierre de vela. */
import { useEffect, useState } from "react";
import useSWR from "swr";
import { ArrowUp, ArrowDown } from "lucide-react";
import {
  endpoints,
  fetcher,
  type AccuracyMetrics,
  type CandleWithChange,
  type Prediction,
} from "@/lib/api";
import { formatAccuracy, formatCountdown, parseIntervalToMs, cn } from "@/lib/utils";
import { usePredictionStore } from "@/store/usePredictionStore";
import { DIRECTION_MODEL_STATUS } from "@/lib/modelStatus";
import { ConfidenceGauge } from "./ConfidenceGauge";
import { Card } from "@/components/ui/Card";
import { CardSkeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";

export function CurrentPredictionCard() {
  const { data, error, isLoading } = useSWR<Prediction>(
    endpoints.predictionCurrent,
    fetcher,
    { refreshInterval: 30_000 },
  );
  const { data: latestCandle } = useSWR<CandleWithChange>(endpoints.candlesLatest, fetcher, {
    refreshInterval: 60_000,
  });
  const { data: accuracy } = useSWR<AccuracyMetrics>(endpoints.metricsAccuracy, fetcher, {
    refreshInterval: 5 * 60_000,
  });
  const livePrediction = usePredictionStore((s) => s.livePrediction);
  const prediction = livePrediction ?? data;

  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  if (isLoading) {
    return (
      <Card title="Predicción Actual">
        <CardSkeleton lines={2} />
      </Card>
    );
  }
  if (error || !prediction) {
    return (
      <Card title="Predicción Actual">
        <ErrorState message="No hay predicción activa disponible." />
      </Card>
    );
  }

  const intervalMs = latestCandle ? parseIntervalToMs(latestCandle.interval) : 4 * 60 * 60 * 1000;
  const openMs = new Date(prediction.candle_open_time).getTime();
  // La predicción de la vela cerrada `open_time` es sobre la vela siguiente, que cierra
  // dos intervalos después de la apertura de la vela usada como entrada.
  const nextCloseMs = openMs + 2 * intervalMs;
  const isUp = prediction.direction === "UP";

  return (
    <Card title="Predicción Actual">
      <div className="flex flex-col items-center gap-4 py-2">
        <div
          className={cn(
            "flex items-center gap-2 rounded-full px-4 py-1.5 text-lg font-bold",
            isUp ? "bg-bull-dim text-bull" : "bg-bear-dim text-bear",
          )}
        >
          {isUp ? <ArrowUp className="h-5 w-5" /> : <ArrowDown className="h-5 w-5" />}
          {prediction.direction}
        </div>

        <ConfidenceGauge confidence={prediction.confidence} direction={prediction.direction} />

        <div className="text-center">
          <p className="text-xs uppercase tracking-wide text-gray-500">Próximo cierre</p>
          <p className="font-mono text-base font-semibold text-gray-200">
            {formatCountdown(nextCloseMs, now)}
          </p>
        </div>

        <p className="text-xs text-gray-500">Modelo v{prediction.model_version}</p>

        {(prediction.status ?? DIRECTION_MODEL_STATUS.status) === "experimental" && (
          <p className="rounded border border-yellow-700/60 px-3 py-2 text-center text-xs text-yellow-300">
            Experimental — no supera a reglas simples.{" "}
            {accuracy?.overall != null && accuracy.baseline_overall != null
              ? `Acierto histórico ${formatAccuracy(accuracy.overall)} frente a ${formatAccuracy(
                  accuracy.baseline_overall,
                )} de la regla «siempre la dirección más frecuente» (${accuracy.total_resolved} predicciones).`
              : `En la investigación: ${(DIRECTION_MODEL_STATUS.accuracy * 100).toFixed(
                  1,
                )}% frente a ${(DIRECTION_MODEL_STATUS.baselineAccuracy * 100).toFixed(1)}% (${
                  DIRECTION_MODEL_STATUS.baselineName
                }).`}
          </p>
        )}
      </div>
    </Card>
  );
}

