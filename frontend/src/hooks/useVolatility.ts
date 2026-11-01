"use client";

import useSWR from "swr";
import {
  endpoints,
  fetcher,
  isNotFound,
  type Horizon,
  type VolatilityForecast,
  type VolatilityMetrics,
} from "@/lib/api";

const REFRESH_MS = 60_000;

/** Un 404 es un estado esperado (volatilidad deshabilitada o sin pronósticos): no se reintenta. */
const swrOptions = {
  refreshInterval: REFRESH_MS,
  shouldRetryOnError: (error: unknown) => !isNotFound(error),
};

export function useVolatility(horizon: Horizon) {
  const { data, error, isLoading } = useSWR<VolatilityForecast>(
    endpoints.volatilityCurrent(horizon),
    fetcher,
    swrOptions,
  );
  return { forecast: data, error, isLoading, unavailable: isNotFound(error) };
}

export function useVolatilityMetrics(horizon: Horizon) {
  const { data, error, isLoading } = useSWR<VolatilityMetrics>(
    endpoints.volatilityMetrics(horizon),
    fetcher,
    { ...swrOptions, refreshInterval: 5 * REFRESH_MS },
  );
  return { metrics: data, error, isLoading, unavailable: isNotFound(error) };
}
