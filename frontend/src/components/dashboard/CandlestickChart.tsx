"use client";

/**
 * Gráfico de velas (TradingView Lightweight Charts) con overlay de markers
 * de predicciones (▲ verde UP / ▼ rojo DOWN), opacidad proporcional a la confianza, y
 * el rango de precio probable a 7 días (bandas 68% y 95%) proyectado desde la última vela.
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
  type LineData,
  ColorType,
  CrosshairMode,
  LineStyle,
} from "lightweight-charts";
import {
  endpoints,
  fetcher,
  isNotFound,
  type Page,
  type Candle,
  type Prediction,
  type PriceBands,
  type VolatilityForecast,
} from "@/lib/api";
import { useMarketStore } from "@/store/useMarketStore";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";

const CANDLE_LIMIT = 300;
const HISTORY_LIMIT = 200;
const DAY_SECONDS = 24 * 3600;
const BAND_HORIZON_SECONDS = 7 * DAY_SECONDS;
const CANDLE_STEP_SECONDS = 4 * 3600;
// Ventana visible al dibujar las bandas: 3 semanas de historia más el horizonte proyectado.
const VISIBLE_HISTORY_SECONDS = 21 * DAY_SECONDS;
// Bandas de volatilidad: colores neutros (no son una señal direccional).
const BAND_LINES: { key: "p68" | "p95"; color: string; style: LineStyle }[] = [
  { key: "p68", color: "#818cf8", style: LineStyle.Dashed },
  { key: "p95", color: "#94a3b8", style: LineStyle.Dotted },
];

/**
 * Cono desde el precio de referencia hasta el borde de la banda al final del horizonte.
 * El eje de tiempo del gráfico es por índice (una barra por punto), así que se emite un punto
 * por vela de 4h; la banda crece con la raíz del tiempo: P·exp(ln(borde/P)·√(i/N)).
 */
function bandCone(
  forecast: VolatilityForecast,
  bands: PriceBands,
  key: "p68" | "p95",
  edge: "low" | "high",
): LineData[] {
  const start = Math.floor(new Date(forecast.candle_open_time).getTime() / 1000);
  const steps = BAND_HORIZON_SECONDS / CANDLE_STEP_SECONDS;
  const logEdge = Math.log(bands[key][edge] / forecast.reference_price);
  return Array.from({ length: steps + 1 }, (_, i) => ({
    time: (start + i * CANDLE_STEP_SECONDS) as unknown as Time,
    value: forecast.reference_price * Math.exp(logEdge * Math.sqrt(i / steps)),
  }));
}

function toUnixSeconds(iso: string): Time {
  return Math.floor(new Date(iso).getTime() / 1000) as unknown as Time;
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
  const { data: volatility } = useSWR<VolatilityForecast>(
    endpoints.volatilityCurrent("7d"),
    fetcher,
    { refreshInterval: 60_000, shouldRetryOnError: (e: unknown) => !isNotFound(e) },
  );
  const bandSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
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
      bandSeriesRef.current = [];
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

  // Rango probable a 7 días (68% y 95%); se redibuja con cada pronóstico nuevo
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    bandSeriesRef.current.forEach((line) => chart.removeSeries(line));
    bandSeriesRef.current = [];
    const bands = volatility?.bands;
    if (!volatility || !bands) return;
    for (const { key, color, style } of BAND_LINES) {
      for (const edge of ["high", "low"] as const) {
        const line = chart.addLineSeries({
          color,
          lineWidth: 1,
          lineStyle: style,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        });
        line.setData(bandCone(volatility, bands, key, edge));
        bandSeriesRef.current.push(line);
      }
    }
    const start = Math.floor(new Date(volatility.candle_open_time).getTime() / 1000);
    chart.timeScale().setVisibleRange({
      from: (start - VISIBLE_HISTORY_SECONDS) as unknown as Time,
      to: (start + BAND_HORIZON_SECONDS + DAY_SECONDS) as unknown as Time,
    });
  }, [volatility]);

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
      {/* El contenedor debe existir desde el primer render: el chart se crea una sola vez
          al montar y necesita el nodo; skeleton y error se superponen en lugar de reemplazarlo. */}
      <div className="relative h-[420px] w-full">
        <div ref={containerRef} className="absolute inset-0" />
        {candlesLoading ? (
          <Skeleton className="absolute inset-0 h-full w-full" />
        ) : candlesError ? (
          <div className="absolute inset-0 bg-bg-surface">
            <ErrorState message="No se pudieron cargar las velas." />
          </div>
        ) : null}
      </div>
      {volatility?.bands && (
        <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-gray-500">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-5 border-t border-dashed border-brand" />
            Rango probable 7d, 68%
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-5 border-t border-dotted border-slate-400" />
            Rango probable 7d, 95%
          </span>
          <span>Describe el riesgo esperado; no indica dirección.</span>
        </p>
      )}
    </Card>
  );
}

