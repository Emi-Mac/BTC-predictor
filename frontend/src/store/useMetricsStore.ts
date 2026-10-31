/**
 * Store global de métricas: detecta model drift (accuracy rolling por debajo
 * del baseline) para mostrar una alerta visual en el dashboard (Fase 6.3).
 */
import { create } from "zustand";

interface MetricsState {
  driftAlert: boolean;
  setDriftAlert: (value: boolean) => void;
}

export const useMetricsStore = create<MetricsState>((set) => ({
  driftAlert: false,
  setDriftAlert: (value) => set({ driftAlert: value }),
}));

