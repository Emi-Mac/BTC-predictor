/**
 * Store de conexión en tiempo real: estado del WebSocket `/ws/live`.
 * Usado por el indicador visual del header y por el hook de reconexión.
 */
import { create } from "zustand";

export type ConnectionStatus = "connecting" | "open" | "reconnecting" | "polling" | "closed";

interface ConnectionState {
  status: ConnectionStatus;
  lastMessageAt: number | null;
  reconnectAttempts: number;
  setStatus: (status: ConnectionStatus) => void;
  markMessageReceived: () => void;
  incrementAttempts: () => void;
  resetAttempts: () => void;
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  status: "connecting",
  lastMessageAt: null,
  reconnectAttempts: 0,
  setStatus: (status) => set({ status }),
  markMessageReceived: () => set({ lastMessageAt: Date.now() }),
  incrementAttempts: () => set((s) => ({ reconnectAttempts: s.reconnectAttempts + 1 })),
  resetAttempts: () => set({ reconnectAttempts: 0 }),
}));

