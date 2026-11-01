/**
 * Store global de predicciones: última predicción recibida vía WebSocket,
 * usada para destacar nuevas predicciones sin esperar la revalidación de SWR.
 */
import { create } from "zustand";
import type { Prediction } from "@/lib/api";

interface PredictionState {
  livePrediction: Prediction | null;
  setLivePrediction: (prediction: Prediction) => void;
}

export const usePredictionStore = create<PredictionState>((set) => ({
  livePrediction: null,
  setLivePrediction: (prediction) => set({ livePrediction: prediction }),
}));

