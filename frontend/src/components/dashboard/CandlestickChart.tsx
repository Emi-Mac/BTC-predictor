"use client";

/**
 * Gráfico de velas (TradingView Lightweight Charts) con overlay de markers
 * de predicciones (▲ verde UP / ▼ rojo DOWN), opacidad proporcional a la confianza.
 */
import { useEffect, useRef } from "react";
import useSWR from "swr";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type SeriesMarker,
  type Time,
  ColorType,
  CrosshairMode,
} from "lightweight-charts";
import { endpoints, fetcher, type Page, type Candle, type Prediction } from "@/lib/api";
import { useMarketStore } from "@/store/useMarketStore";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";

const CANDLE_LIMIT = 300;
const HISTORY_LIMIT = 200;

function toUnixSeconds(iso: string): Time {
  return (Math.floor(new Date(iso).getTime() / 1000) as unknown) as Time;
}

export function CandlestickChart() {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  const { data: candlesPage, error: candlesError, isLoading: candlesLoading } = useSWR<
    Page<Candle>
  >(endpoints.candles(CANDLE_LIMIT, 0), fetcher, { refreshInterval: 60_000 });
  const { data: historyPage } = useSWR<Page<Prediction>>(
    endpoints.predictionHistory(HISTORY_LIMIT, 0),
    fetcher,
    { refreshInterval: 60_000 },
  );
  const liveCandle = useMarketStore((s) => s.liveCandle);

  // Inicialización del chart (una sola vez)
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#111827" },
        textColor: "#9ca3af",
        fontFamily: "JetBrains Mono, monospace",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#374151" },
      timeScale: { borderColor: "#374151", timeVisible: true, secondsVisible: false },
      autoSize: true,
    });
    const series = chart.addCandlestickSeries({
      upColor: "#4ade80",
      downColor: "#f87171",
      borderUpColor: "#4ade80",
      borderDownColor: "#f87171",
      wickUpColor: "#4ade80",
      wickDownColor: "#f87171",
    });
    chartRef.current = chart;
    seriesRef.current = series;

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  // Carga de datos de velas (solo `setData`, una vez por fetch)
  useEffect(() => {
    if (!seriesRef.current || !candlesPage?.data?.length) return;
    const sorted = [...candlesPage.data].sort(
      (a, b) => new Date(a.open_time).getTime() - new Date(b.open_time).getTime(),
    );
    const bars: CandlestickData[] = sorted.map((c) => ({
      time: toUnixSeconds(c.open_time),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    seriesRef.current.setData(bars);
    chartRef.current?.timeScale().fitContent();
  }, [candlesPage]);

  // Update incremental con la vela en vivo (WebSocket)
  useEffect(() => {
    if (!seriesRef.current || !liveCandle) return;
    seriesRef.current.update({
      time: toUnixSeconds(liveCandle.open_time),
      open: liveCandle.open,
      high: liveCandle.high,
      low: liveCandle.low,
      close: liveCandle.close,
    });
  }, [liveCandle]);

  // Markers de predicciones sobre el gráfico
  useEffect(() => {
    if (!seriesRef.current || !historyPage?.data?.length) return;
    const markers: SeriesMarker<Time>[] = historyPage.data
      .slice()
      .sort((a, b) => new Date(a.candle_open_time).getTime() - new Date(b.candle_open_time).getTime())
      .map((p) => ({
        time: toUnixSeconds(p.candle_open_time),
        position: p.direction === "UP" ? "belowBar" : "aboveBar",
        color: p.direction === "UP" ? "#4ade80" : "#f87171",
        shape: p.direction === "UP" ? "arrowUp" : "arrowDown",
        text: `${(p.confidence * 100).toFixed(0)}%`,
      }));
    seriesRef.current.setMarkers(markers);
  }, [historyPage]);

  return (
    <Card title="Gráfico BTC/USDT" className="h-full">
      {candlesLoading ? (
        <Skeleton className="h-[420px] w-full" />
      ) : candlesError ? (
        <ErrorState message="No se pudieron cargar las velas." />
      ) : (
        <div ref={containerRef} className="h-[420px] w-full" />
      )}
    </Card>
  );
}

