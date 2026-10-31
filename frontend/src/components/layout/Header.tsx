"use client";

/** Header principal: logo, precio en vivo, variación 24h, estado de conexión y badge del modelo. */
import useSWR from "swr";
import { TrendingUp, TrendingDown, Bitcoin } from "lucide-react";
import { endpoints, fetcher, type CandleWithChange, type ModelMetadata } from "@/lib/api";
import { formatCurrency, formatPercent, formatAccuracy, cn } from "@/lib/utils";
import { useMarketStore } from "@/store/useMarketStore";
import { ConnectionStatus } from "./ConnectionStatus";
import { Skeleton } from "@/components/ui/Skeleton";

export function Header() {
  const { data: candle, isLoading } = useSWR<CandleWithChange>(endpoints.candlesLatest, fetcher, {
    refreshInterval: 60_000,
  });
  const { data: model } = useSWR<ModelMetadata>(endpoints.metricsModel, fetcher, {
    refreshInterval: 5 * 60_000,
  });
  const liveCandle = useMarketStore((s) => s.liveCandle);

  const effective = liveCandle ?? candle;
  const isUp = (effective?.change_pct ?? 0) >= 0;

  return (
    <header className="flex items-center justify-between gap-4 border-b border-border bg-bg-primary/90 px-4 py-3 backdrop-blur-sm sm:px-6">
      <div className="flex items-center gap-2">
        <Bitcoin className="h-6 w-6 text-brand" />
        <span className="hidden text-lg font-bold tracking-tight sm:inline">BTC Predictor</span>
      </div>

      <div className="flex min-w-0 items-center gap-3 font-mono">
        {isLoading && !effective ? (
          <Skeleton className="h-6 w-32" />
        ) : effective ? (
          <>
            <span className="text-lg font-semibold text-gray-50 sm:text-xl">
              {formatCurrency(effective.close)}
            </span>
            <span
              className={cn(
                "flex items-center gap-1 text-sm font-semibold",
                isUp ? "text-bull" : "text-bear",
              )}
            >
              {isUp ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
              {formatPercent(effective.change_pct)}
            </span>
          </>
        ) : (
          <span className="text-sm text-gray-500">Sin datos</span>
        )}
      </div>

      <div className="flex items-center gap-4">
        {model && (
          <span className="hidden items-center gap-1 rounded-full bg-bg-surface px-3 py-1 text-xs text-gray-300 md:inline-flex">
            v{model.version} · {formatAccuracy(model.val_accuracy)} acc
          </span>
        )}
        <ConnectionStatus />
      </div>
    </header>
  );
}


