"use client";

/** Tabla paginada del historial de predicciones, filtrable por dirección y acierto. */
import { useState } from "react";
import useSWR from "swr";
import { ChevronLeft, ChevronRight, Check, X, Minus } from "lucide-react";
import { endpoints, fetcher, type Page, type Prediction } from "@/lib/api";
import { formatDateTime, cn } from "@/lib/utils";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { ErrorState } from "@/components/ui/ErrorState";

const PAGE_SIZE = 10;

type DirectionFilter = "ALL" | "UP" | "DOWN";
type ResultFilter = "ALL" | "CORRECT" | "INCORRECT";

export function PredictionHistoryTable() {
  const [offset, setOffset] = useState(0);
  const [directionFilter, setDirectionFilter] = useState<DirectionFilter>("ALL");
  const [resultFilter, setResultFilter] = useState<ResultFilter>("ALL");

  const { data, error, isLoading } = useSWR<Page<Prediction>>(
    endpoints.predictionHistory(PAGE_SIZE, offset),
    fetcher,
    { refreshInterval: 60_000 },
  );

  const rows = (data?.data ?? []).filter((p) => {
    if (directionFilter !== "ALL" && p.direction !== directionFilter) return false;
    if (resultFilter === "CORRECT" && p.is_correct !== true) return false;
    if (resultFilter === "INCORRECT" && p.is_correct !== false) return false;
    return true;
  });

  const pagination = data?.pagination;

  return (
    <Card
      title="Historial de Predicciones"
      action={
        <div className="flex gap-2 text-xs">
          <select
            value={directionFilter}
            onChange={(e) => setDirectionFilter(e.target.value as DirectionFilter)}
            className="rounded-md border border-border bg-bg-elevated px-2 py-1 text-gray-300"
            aria-label="Filtrar por dirección"
          >
            <option value="ALL">Todas</option>
            <option value="UP">Solo UP</option>
            <option value="DOWN">Solo DOWN</option>
          </select>
          <select
            value={resultFilter}
            onChange={(e) => setResultFilter(e.target.value as ResultFilter)}
            className="rounded-md border border-border bg-bg-elevated px-2 py-1 text-gray-300"
            aria-label="Filtrar por resultado"
          >
            <option value="ALL">Todos</option>
            <option value="CORRECT">Aciertos</option>
            <option value="INCORRECT">Fallos</option>
          </select>
        </div>
      }
    >
      {isLoading ? (
        <div className="flex flex-col gap-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-8 w-full" />
          ))}
        </div>
      ) : error ? (
        <ErrorState message="No se pudo cargar el historial." />
      ) : (
        <>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[480px] text-left text-sm">
              <thead>
                <tr className="border-b border-border text-xs uppercase text-gray-500">
                  <th className="py-2 pr-2 font-medium">Hora</th>
                  <th className="py-2 pr-2 font-medium">Pred.</th>
                  <th className="py-2 pr-2 font-medium">Conf.</th>
                  <th className="py-2 pr-2 font-medium">Real</th>
                  <th className="py-2 font-medium">OK</th>
                </tr>
              </thead>
              <tbody>
                {rows.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-4 text-center text-gray-500">
                      Sin resultados para este filtro.
                    </td>
                  </tr>
                ) : (
                  rows.map((p) => (
                    <tr key={p.id} className="border-b border-border/60 last:border-0">
                      <td className="py-2 pr-2 font-mono text-gray-400">
                        {formatDateTime(p.predicted_at)}
                      </td>
                      <td className="py-2 pr-2">
                        <span className={p.direction === "UP" ? "badge-up" : "badge-down"}>
                          {p.direction === "UP" ? "▲" : "▼"} {p.direction}
                        </span>
                      </td>
                      <td className="py-2 pr-2 font-mono text-gray-300">
                        {(p.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="py-2 pr-2">
                        {p.actual_direction ? (
                          <span className={p.actual_direction === "UP" ? "text-bull" : "text-bear"}>
                            {p.actual_direction === "UP" ? "▲" : "▼"}
                          </span>
                        ) : (
                          <span className="text-gray-600">—</span>
                        )}
                      </td>
                      <td className="py-2">
                        {p.is_correct === true && <Check className="h-4 w-4 text-bull" />}
                        {p.is_correct === false && <X className="h-4 w-4 text-bear" />}
                        {p.is_correct === null && <Minus className="h-4 w-4 text-gray-600" />}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {pagination && (
            <div className="flex items-center justify-between pt-2 text-xs text-gray-400">
              <span>
                Mostrando {pagination.offset + 1}–
                {Math.min(pagination.offset + PAGE_SIZE, pagination.total)} de {pagination.total}
              </span>
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                  disabled={offset === 0}
                  className={cn(
                    "rounded-md border border-border p-1.5 hover:bg-bg-elevated",
                    offset === 0 && "cursor-not-allowed opacity-40",
                  )}
                  aria-label="Página anterior"
                >
                  <ChevronLeft className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => setOffset((o) => o + PAGE_SIZE)}
                  disabled={!pagination.has_more}
                  className={cn(
                    "rounded-md border border-border p-1.5 hover:bg-bg-elevated",
                    !pagination.has_more && "cursor-not-allowed opacity-40",
                  )}
                  aria-label="Página siguiente"
                >
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </Card>
  );
}

