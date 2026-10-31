/** Gauge semicircular de confianza: 50%=neutral, 60%+=verde (UP) / rojo (DOWN). */
import { cn } from "@/lib/utils";
import type { Direction } from "@/lib/api";

interface Props {
  confidence: number; // 0..1
  direction: Direction;
}

const SIZE = 160;
const STROKE = 14;
const RADIUS = SIZE / 2 - STROKE;
const CIRCUMFERENCE = Math.PI * RADIUS; // semicírculo

export function ConfidenceGauge({ confidence, direction }: Props) {
  const pct = Math.min(1, Math.max(0, confidence));
  const dashOffset = CIRCUMFERENCE * (1 - pct);
  const colorClass = direction === "UP" ? "stroke-bull" : "stroke-bear";

  return (
    <div className="flex flex-col items-center">
      <svg width={SIZE} height={SIZE / 2 + STROKE} viewBox={`0 0 ${SIZE} ${SIZE / 2 + STROKE}`}>
        {/* Fondo del gauge */}
        <path
          d={`M ${STROKE} ${SIZE / 2} A ${RADIUS} ${RADIUS} 0 0 1 ${SIZE - STROKE} ${SIZE / 2}`}
          fill="none"
          strokeWidth={STROKE}
          strokeLinecap="round"
          className="stroke-bg-elevated"
        />
        {/* Progreso según confianza */}
        <path
          d={`M ${STROKE} ${SIZE / 2} A ${RADIUS} ${RADIUS} 0 0 1 ${SIZE - STROKE} ${SIZE / 2}`}
          fill="none"
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={dashOffset}
          className={cn("transition-all duration-700 ease-out", colorClass)}
        />
      </svg>
      <span
        className={cn(
          "-mt-8 font-mono text-2xl font-bold",
          direction === "UP" ? "text-bull" : "text-bear",
        )}
      >
        {(pct * 100).toFixed(1)}%
      </span>
      <span className="text-xs text-gray-500">confianza del modelo</span>
    </div>
  );
}

