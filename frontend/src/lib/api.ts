/**
 * Cliente HTTP centralizado para comunicarse con el backend FastAPI.
 * Tipos alineados 1:1 con las entidades de dominio del backend (Fase 4).
 * Usa SWR para cache y revalidación automática.
 */
import axios from "axios";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 10_000, // 10 segundos
});

// Interceptor para logging en desarrollo
if (process.env.NODE_ENV === "development") {
  apiClient.interceptors.request.use((config) => {
    console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  });
}

// Interceptor para manejo centralizado de errores
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error)) {
      console.error(`[API Error] ${error.response?.status}: ${error.message}`);
    }
    return Promise.reject(error);
  },
);

// ── SWR fetcher genérico ──────────────────────────────────────
export const fetcher = <T>(url: string): Promise<T> =>
  apiClient.get<T>(url).then((res) => res.data);

// ── Paginación genérica ─────────────────────────────────────────
export interface PaginationMeta {
  limit: number;
  offset: number;
  total: number;
  has_more: boolean;
}

export interface Page<T> {
  data: T[];
  pagination: PaginationMeta;
}

// ── Tipos de respuesta de la API (alineados con el backend) ───
export interface Candle {
  id: number;
  symbol: string;
  interval: string;
  open_time: string;
  close_time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  quote_volume: number | null;
  num_trades: number | null;
}

export interface CandleWithChange extends Candle {
  change_pct: number | null;
}

export type Direction = "UP" | "DOWN";

export interface Prediction {
  id: number;
  candle_id: number;
  candle_open_time: string;
  model_version: number;
  direction: Direction;
  confidence: number;
  probability_up: number | null;
  predicted_at: string;
  actual_direction: Direction | null;
  is_correct: boolean | null;
  resolved_at: string | null;
  /** Ausente hasta que el backend lo exponga (F4); la UI asume «experimental». */
  status?: ModelStatus;
}

export interface MonthlyAccuracy {
  month: string; // "YYYY-MM"
  accuracy: number | null;
  total_predictions: number;
}

export interface AccuracyMetrics {
  overall: number | null;
  last_30d: number | null;
  last_7d: number | null;
  total_resolved: number;
  /** Accuracy del mejor baseline en el mismo período; mostrar junto a `overall`. */
  baseline_overall?: number | null;
  by_month: MonthlyAccuracy[];
}

/**
 * "experimental": no supera a los baselines con significancia (gate de docs/research).
 * "validated": superó el gate de desarrollo y un holdout independiente.
 */
export type ModelStatus = "experimental" | "validated";

export interface ModelMetadata {
  version: number;
  model_type: string;
  val_accuracy: number | null;
  val_f1: number | null;
  val_precision: number | null;
  val_recall: number | null;
  baseline_accuracy: number | null;
  best_baseline_accuracy?: number | null;
  beats_baseline?: boolean;
  status?: ModelStatus;
  evidence_url?: string | null;
  trained_at: string;
  is_active: boolean;
  feature_names: string[];
  artifact_path: string | null;
  scaler_path: string | null;
}

export interface FeatureSnapshot {
  candle_id: number;
  symbol: string;
  interval: string;
  open_time: string;
  values: Record<string, number | boolean | null>;
}

// ── Volatilidad y riesgo (no son una señal direccional) ───────────
export type Horizon = "1d" | "7d";
export type VolatilityRegime = "low" | "normal" | "high" | "extreme";
/** «in_validation» hasta completar la sombra prospectiva de 12 semanas. */
export type ValidationStatus = "in_validation" | "validated";

export interface PriceBand {
  low: number;
  high: number;
}

export interface PriceBands {
  p68: PriceBand;
  p90: PriceBand;
  p95: PriceBand;
}

export interface VolatilityForecast {
  horizon: Horizon;
  as_of: string;
  candle_open_time: string;
  reference_price: number;
  vol_annual: number;
  /** Desviación estándar del retorno a lo largo del horizonte, en %. */
  vol_horizon_pct: number;
  regime: VolatilityRegime;
  /** `null` mientras no haya calibración suficiente. */
  bands: PriceBands | null;
  dvol: number | null;
  realized_vol_7d: number | null;
  realized_vol_30d: number | null;
  model_version: number;
  /** Motivo por el que se sirvió el modelo de respaldo, o `null` si fue el primario. */
  fallback: string | null;
  validation_status: ValidationStatus;
}

export interface VolatilityHistoryPoint {
  candle_open_time: string;
  as_of: string;
  forecast_vol_annual: number;
  realized_vol_annual: number | null;
  regime: VolatilityRegime;
  model_version: number;
  fallback: string | null;
  resolved: boolean;
  inside_68: boolean | null;
  inside_90: boolean | null;
  inside_95: boolean | null;
}

export interface WindowCoverage {
  weeks: number;
  resolved: number;
  coverage_68: number | null;
  coverage_90: number | null;
  coverage_95: number | null;
}

export interface VolatilityMetrics {
  horizon: Horizon;
  resolved: number;
  coverage_68: number | null;
  coverage_90: number | null;
  coverage_95: number | null;
  qlike: number | null;
  qlike_naive: number | null;
  mz_slope: number | null;
  last_12w: WindowCoverage;
  validation_status: ValidationStatus;
}

export type Side = "long" | "short";

export interface PositionSize {
  horizon: Horizon;
  side: Side;
  reference_price: number;
  vol_horizon_pct: number;
  stop_distance_pct: number;
  stop_price: number;
  risk_usd: number;
  position_usd: number;
  position_btc: number;
  exposure_pct: number;
  regime: VolatilityRegime;
  warnings: string[];
}

export interface PositionSizeParams {
  capital: number;
  riskPct: number;
  stopK: number;
  horizon: Horizon;
  side: Side;
}

// ── Endpoints (las claves también se usan como keys de SWR) ───────
export const endpoints = {
  candles: (limit = 200, offset = 0) => `/api/v1/candles?limit=${limit}&offset=${offset}`,
  candlesLatest: "/api/v1/candles/latest",
  predictionCurrent: "/api/v1/predictions/current",
  predictionHistory: (limit = 20, offset = 0) =>
    `/api/v1/predictions/history?limit=${limit}&offset=${offset}`,
  metricsAccuracy: "/api/v1/metrics/accuracy",
  metricsModel: "/api/v1/metrics/model",
  featuresLatest: "/api/v1/features/latest",
  volatilityCurrent: (horizon: Horizon) => `/api/v1/volatility/current?horizon=${horizon}`,
  volatilityHistory: (horizon: Horizon, limit = 200, offset = 0) =>
    `/api/v1/volatility/history?horizon=${horizon}&limit=${limit}&offset=${offset}`,
  volatilityMetrics: (horizon: Horizon) => `/api/v1/metrics/volatility?horizon=${horizon}`,
  positionSize: (p: PositionSizeParams) =>
    `/api/v1/risk/position-size?capital=${p.capital}&risk_pct=${p.riskPct}` +
    `&stop_k=${p.stopK}&horizon=${p.horizon}&side=${p.side}`,
} as const;

/** Prefijo común de las claves SWR de volatilidad (para invalidarlas juntas). */
export const VOLATILITY_KEY_PREFIXES = ["/api/v1/volatility/", "/api/v1/metrics/volatility"];

/** `true` si el backend respondió 404 (volatilidad deshabilitada o sin pronósticos aún). */
export function isNotFound(error: unknown): boolean {
  return axios.isAxiosError(error) && error.response?.status === 404;
}

// ── WebSocket ──────────────────────────────────────────────────────
const WS_BASE = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";
export const WS_LIVE_URL = `${WS_BASE}/ws/live`;

export interface WsMessage {
  type: "heartbeat" | "candle" | "prediction" | "volatility" | string;
  data?: unknown;
}


