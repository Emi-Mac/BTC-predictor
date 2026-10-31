"use client";

/**
 * Hook de conexión WebSocket a `/ws/live` con:
 *  - Heartbeat: si no llega ningún mensaje en 45s se asume conexión muerta.
 *  - Reconexión automática con backoff exponencial + jitter (1s..30s).
 *  - Fallback a polling HTTP si falla 3 veces consecutivas (señaliza el
 *    status "polling" para que los componentes aumenten su refreshInterval de SWR).
 *  - Invalida las claves SWR relevantes cuando llegan eventos "candle" / "prediction".
 */
import { useEffect, useRef } from "react";
import { mutate } from "swr";
import { endpoints, WS_LIVE_URL, type CandleWithChange, type Prediction } from "@/lib/api";
import { backoffDelayMs } from "@/lib/utils";
import { useConnectionStore } from "@/store/connectionStore";
import { useMarketStore } from "@/store/useMarketStore";
import { usePredictionStore } from "@/store/usePredictionStore";
import { useToastStore } from "@/store/toastStore";

const HEARTBEAT_TIMEOUT_MS = 45_000;
const MAX_CONSECUTIVE_FAILURES = 3;

export function useLiveConnection(): void {
  const consecutiveFailures = useRef(0);
  const heartbeatTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    const { setStatus, markMessageReceived, incrementAttempts, resetAttempts } =
      useConnectionStore.getState();

    const clearHeartbeatTimer = () => {
      if (heartbeatTimer.current) clearTimeout(heartbeatTimer.current);
    };

    const scheduleHeartbeatCheck = () => {
      clearHeartbeatTimer();
      heartbeatTimer.current = setTimeout(() => {
        // No llegó ningún mensaje (ni heartbeat) a tiempo: forzar reconexión.
        socketRef.current?.close();
      }, HEARTBEAT_TIMEOUT_MS);
    };

    const connect = () => {
      if (!mountedRef.current) return;
      setStatus(consecutiveFailures.current > 0 ? "reconnecting" : "connecting");

      let socket: WebSocket;
      try {
        socket = new WebSocket(WS_LIVE_URL);
      } catch {
        scheduleReconnect();
        return;
      }
      socketRef.current = socket;

      socket.onopen = () => {
        consecutiveFailures.current = 0;
        resetAttempts();
        setStatus("open");
        scheduleHeartbeatCheck();
      };

      socket.onmessage = (event) => {
        markMessageReceived();
        scheduleHeartbeatCheck();
        try {
          const message = JSON.parse(event.data as string) as {
            type: string;
            data?: unknown;
          };
          handleMessage(message);
        } catch {
          // Mensaje no-JSON: ignorar silenciosamente.
        }
      };

      socket.onclose = () => {
        clearHeartbeatTimer();
        if (!mountedRef.current) return;
        consecutiveFailures.current += 1;
        incrementAttempts();
        scheduleReconnect();
      };

      socket.onerror = () => {
        socket.close();
      };
    };

    const scheduleReconnect = () => {
      const status = consecutiveFailures.current >= MAX_CONSECUTIVE_FAILURES ? "polling" : "reconnecting";
      useConnectionStore.getState().setStatus(status);
      const delay = backoffDelayMs(consecutiveFailures.current);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      reconnectTimer.current = setTimeout(connect, delay);
    };

    const handleMessage = (message: { type: string; data?: unknown }) => {
      switch (message.type) {
        case "candle":
          if (message.data) {
            useMarketStore.getState().setLiveCandle(message.data as CandleWithChange);
            void mutate(endpoints.candlesLatest);
            void mutate((key: string) => key.startsWith("/api/v1/candles?"));
          }
          break;
        case "prediction":
          if (message.data) {
            const prediction = message.data as Prediction;
            usePredictionStore.getState().setLivePrediction(prediction);
            useToastStore
              .getState()
              .addToast(
                `Nueva predicción: ${prediction.direction} (${(prediction.confidence * 100).toFixed(1)}%)`,
                "info",
              );
            void mutate(endpoints.predictionCurrent);
            void mutate((key: string) => key.startsWith("/api/v1/predictions/history"));
          }
          break;
        case "heartbeat":
        default:
          break;
      }
    };

    connect();

    return () => {
      mountedRef.current = false;
      clearHeartbeatTimer();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      socketRef.current?.close();
    };
  }, []);
}

