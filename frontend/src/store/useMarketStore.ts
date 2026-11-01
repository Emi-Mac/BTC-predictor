/**
 * Store global del mercado: sobreescribe el precio/vela "en vivo" recibido
 * por WebSocket por encima de lo cacheado por SWR, para feedback instantáneo.
 */
import { create } from "zustand";
import type { CandleWithChange } from "@/lib/api";

interface MarketState {
  liveCandle: CandleWithChange | null;
  setLiveCandle: (candle: CandleWithChange) => void;
}

export const useMarketStore = create<MarketState>((set) => ({
  liveCandle: null,
  setLiveCandle: (candle) => set({ liveCandle: candle }),
}));

