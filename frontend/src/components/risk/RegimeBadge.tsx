import type { VolatilityRegime } from "@/lib/api";
import { REGIME_CLASS, REGIME_LABEL } from "@/lib/volatility";
import { cn } from "@/lib/utils";

/** Régimen de riesgo con texto explícito; descriptivo, no una señal de compra o venta. */
export function RegimeBadge({ regime }: { regime: VolatilityRegime }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold",
        REGIME_CLASS[regime],
      )}
    >
      Régimen {REGIME_LABEL[regime].toLowerCase()}
    </span>
  );
}
