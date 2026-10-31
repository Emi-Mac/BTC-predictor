/**
 * Utilidades compartidas: merge de clases Tailwind, formateadores financieros
 * y helpers de fechas/intervalos usados en todo el dashboard.
 */
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Combina clases condicionalmente y resuelve conflictos de Tailwind. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Formatea un número como precio USD (BTC/USDT). */
export function formatCurrency(value: number, maximumFractionDigits = 2): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits,
  }).format(value);
}

/** Formatea un porcentaje con signo (+/-) y 2 decimales. */
export function formatPercent(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}%`;
}

/** Formatea un accuracy 0-1 (o 0-100) como porcentaje legible. */
export function formatAccuracy(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const pct = value <= 1 ? value * 100 : value;
  return `${pct.toFixed(1)}%`;
}

/** Formatea una fecha ISO como hora local corta (HH:mm). */
export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString("es-ES", { hour: "2-digit", minute: "2-digit" });
}

/** Formatea una fecha ISO como fecha+hora corta. */
export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("es-ES", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Convierte un intervalo estilo Binance ("4h", "1d", "15m") a milisegundos. */
export function parseIntervalToMs(interval: string): number {
  const match = /^(\d+)([mhdw])$/.exec(interval.trim().toLowerCase());
  if (!match) return 4 * 60 * 60 * 1000; // fallback: 4h
  const amount = Number(match[1]);
  const unitMs: Record<string, number> = {
    m: 60_000,
    h: 60 * 60_000,
    d: 24 * 60 * 60_000,
    w: 7 * 24 * 60 * 60_000,
  };
  return amount * (unitMs[match[2]] ?? unitMs.h);
}

/** Devuelve un string "Hh Mm Ss" con el tiempo restante hasta `targetMs`. */
export function formatCountdown(targetMs: number, nowMs: number): string {
  const diff = Math.max(0, targetMs - nowMs);
  const totalSeconds = Math.floor(diff / 1000);
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  return `${h}h ${m.toString().padStart(2, "0")}m ${s.toString().padStart(2, "0")}s`;
}

/** Jitter exponencial para reconexión de WebSocket: 1s,2s,4s,8s,16s,max 30s. */
export function backoffDelayMs(attempt: number): number {
  const base = Math.min(30_000, 1000 * 2 ** attempt);
  const jitter = Math.random() * 0.3 * base;
  return base + jitter;
}

