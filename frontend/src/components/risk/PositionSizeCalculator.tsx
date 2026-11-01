"use client";

/**
 * Calculadora educativa de tamaño de posición: arriesga un % fijo del capital hasta un stop
 * situado a k desviaciones del horizonte. No emite señales ni órdenes.
 */
import { useEffect, useState } from "react";
import axios from "axios";
import useSWR from "swr";
import { AlertTriangle } from "lucide-react";
import {
  endpoints,
  fetcher,
  isNotFound,
  type Horizon,
  type PositionSize,
  type Side,
} from "@/lib/api";
import { HORIZON_LABEL } from "@/lib/volatility";
import { cn, formatCurrency } from "@/lib/utils";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { RegimeBadge } from "./RegimeBadge";

const GUIDE_URL =
  "https://github.com/Emi-Mac/BTC-predictor/blob/feature/prediccion-volatilidad/docs/trading/05-ordenes-y-tamano-de-posicion.md";
const RISK_MIN = 0.25;
const RISK_MAX = 2;
const STOP_OPTIONS = [1, 1.5, 2, 3];
const DEBOUNCE_MS = 400;

function useDebounced<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}

function errorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = (error.response?.data as { detail?: unknown } | undefined)
      ?.detail;
    if (typeof detail === "string") return detail;
    if (error.response?.status === 422) return "Revisa los valores ingresados.";
  }
  return "No se pudo calcular el tamaño de posición.";
}

function Output({
  label,
  value,
  emphasis,
}: {
  label: string;
  value: string;
  emphasis?: boolean;
}) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-gray-400">{label}</span>
      <span
        className={cn(
          "font-mono",
          emphasis ? "text-base font-bold text-gray-100" : "text-gray-200",
        )}
      >
        {value}
      </span>
    </div>
  );
}

export function PositionSizeCalculator() {
  const [capitalText, setCapitalText] = useState("10000");
  const [riskPct, setRiskPct] = useState(1);
  const [stopK, setStopK] = useState(2);
  const [horizon, setHorizon] = useState<Horizon>("1d");
  const [side, setSide] = useState<Side>("long");

  const capital = Number(capitalText);
  const capitalValid = Number.isFinite(capital) && capital > 0;
  const params = useDebounced(
    { capital, riskPct, stopK, horizon, side },
    DEBOUNCE_MS,
  );
  const key =
    capitalValid && params.capital > 0 ? endpoints.positionSize(params) : null;

  const { data, error, isLoading } = useSWR<PositionSize>(key, fetcher, {
    keepPreviousData: true,
    shouldRetryOnError: false,
  });

  return (
    <Card title="Calculadora de posición">
      <p className="text-xs text-gray-500">
        Herramienta educativa: calcula cuánto comprar o vender para que, si el
        precio toca el stop, pierdas solo el % de riesgo elegido. No es asesoría
        financiera.
      </p>

      <div className="grid grid-cols-2 gap-3 text-xs text-gray-400">
        <label className="flex flex-col gap-1">
          Capital (USD)
          <input
            inputMode="decimal"
            value={capitalText}
            onChange={(e) => setCapitalText(e.target.value)}
            aria-invalid={!capitalValid}
            className={cn(
              "rounded-md border bg-bg-elevated px-2 py-1.5 font-mono text-sm text-gray-100 outline-none focus:border-brand",
              capitalValid ? "border-border" : "border-amber-400",
            )}
          />
        </label>
        <label className="flex flex-col gap-1">
          Riesgo por operación:{" "}
          <span className="font-mono text-gray-200">{riskPct}%</span>
          <input
            type="range"
            min={RISK_MIN}
            max={RISK_MAX}
            step={0.25}
            value={riskPct}
            onChange={(e) => setRiskPct(Number(e.target.value))}
            className="accent-brand"
          />
        </label>
        <label className="flex flex-col gap-1">
          Stop (múltiplo de σ)
          <select
            value={stopK}
            onChange={(e) => setStopK(Number(e.target.value))}
            className="rounded-md border border-border bg-bg-elevated px-2 py-1.5 text-sm text-gray-100"
          >
            {STOP_OPTIONS.map((k) => (
              <option key={k} value={k}>
                {k}σ
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          Horizonte del stop
          <select
            value={horizon}
            onChange={(e) => setHorizon(e.target.value as Horizon)}
            className="rounded-md border border-border bg-bg-elevated px-2 py-1.5 text-sm text-gray-100"
          >
            {(["1d", "7d"] as Horizon[]).map((h) => (
              <option key={h} value={h}>
                {HORIZON_LABEL[h]}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div role="group" aria-label="Lado" className="flex gap-1 text-xs">
        {(["long", "short"] as Side[]).map((s) => (
          <button
            key={s}
            type="button"
            aria-pressed={side === s}
            onClick={() => setSide(s)}
            className={cn(
              "flex-1 rounded-md border px-2 py-1",
              side === s
                ? "border-brand bg-brand/20 text-gray-100"
                : "border-border text-gray-400 hover:text-gray-200",
            )}
          >
            {s === "long" ? "Posición larga" : "Posición corta"}
          </button>
        ))}
      </div>

      {!capitalValid ? (
        <p className="text-xs text-amber-300">
          Ingresa un capital mayor que 0.
        </p>
      ) : error ? (
        <div className="flex gap-2 rounded-md border border-amber-400/40 bg-amber-400/10 px-3 py-2 text-xs text-amber-300">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            {isNotFound(error)
              ? "No hay pronóstico de volatilidad disponible."
              : errorMessage(error)}
          </span>
        </div>
      ) : !data ? (
        isLoading ? (
          <Skeleton className="h-28 w-full" />
        ) : null
      ) : (
        <div className={cn("flex flex-col gap-1.5", isLoading && "opacity-60")}>
          <Output
            label="Distancia al stop"
            value={`${data.stop_distance_pct.toFixed(2)}%`}
          />
          <Output
            label="Precio del stop"
            value={formatCurrency(data.stop_price, 0)}
          />
          <Output
            label="Pérdida si se activa"
            value={formatCurrency(data.risk_usd)}
          />
          <Output
            label="Tamaño de posición"
            value={formatCurrency(data.position_usd, 0)}
            emphasis
          />
          <Output
            label="Equivalente en BTC"
            value={`${data.position_btc.toFixed(4)} BTC`}
          />
          <Output
            label="Exposición del capital"
            value={`${data.exposure_pct.toFixed(1)}%`}
          />
          <div className="flex items-center justify-between pt-1 text-xs text-gray-500">
            <span>
              σ a {HORIZON_LABEL[data.horizon]}:{" "}
              {data.vol_horizon_pct.toFixed(2)}%
            </span>
            <RegimeBadge regime={data.regime} />
          </div>
          {data.warnings.length > 0 && (
            <ul className="mt-1 flex flex-col gap-1 text-xs text-amber-300">
              {data.warnings.map((w) => (
                <li key={w} className="flex gap-1.5">
                  <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
                  <span>{w}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <p className="text-[11px] text-gray-500">
        El stop no garantiza la pérdida máxima (puede haber deslizamiento).{" "}
        <a
          href={GUIDE_URL}
          target="_blank"
          rel="noreferrer"
          className="text-brand underline"
        >
          Guía de órdenes y tamaño de posición
        </a>
        .
      </p>
    </Card>
  );
}
