/**
 * Cliente HTTP centralizado para comunicarse con el backend FastAPI.
 * Usa SWR para cache y revalidación automática.
 */
import axios from "axios";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

// ── Tipos de respuesta de la API (se completarán en Fase 4) ───
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
}

export interface Prediction {
  id: number;
  candle_id: number;
  model_version: number;
  predicted_direction: "UP" | "DOWN";
  confidence_score: number;
  raw_probability_up: number | null;
  actual_direction: "UP" | "DOWN" | null;
  is_correct: boolean | null;
  predicted_at: string;
  resolved_at: string | null;
}

export interface AccuracyMetrics {
  overall: number;
  last_7d: number;
  last_30d: number;
  last_90d: number;
  total_predictions: number;
  correct_predictions: number;
}

export interface ModelMetadata {
  version: number;
  model_type: string;
  val_accuracy: number;
  val_f1: number;
  baseline_accuracy: number;
  trained_at: string;
  is_active: boolean;
}

// ── WebSocket URL ──────────────────────────────────────────────
export const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";

