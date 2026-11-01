"use client";

/** Panel de riesgo: agrupa las tarjetas de volatilidad y se oculta si el backend no la expone. */
import { useVolatility } from "@/hooks/useVolatility";
import { Card } from "@/components/ui/Card";
import { CalibrationPanel } from "./CalibrationPanel";
import { ExpectedRangeCard } from "./ExpectedRangeCard";
import { PositionSizeCalculator } from "./PositionSizeCalculator";
import { VolatilityCard } from "./VolatilityCard";
import { VolatilityHistoryChart } from "./VolatilityHistoryChart";

export function RiskPanel() {
  const { unavailable, isLoading } = useVolatility("7d");

  if (!isLoading && unavailable) {
    return (
      <Card title="Panel de riesgo">
        <p className="text-sm text-gray-400">
          El pronóstico de volatilidad no está disponible: el backend lo tiene
          deshabilitado o el modelo aún no emitió pronósticos.
        </p>
      </Card>
    );
  }

  return (
    <section aria-labelledby="risk-panel-title" className="flex flex-col gap-3">
      <div>
        <h2
          id="risk-panel-title"
          className="text-sm font-semibold uppercase tracking-wide text-gray-300"
        >
          Panel de riesgo
        </h2>
        <p className="text-xs text-gray-500">
          La volatilidad sí es predecible, la dirección no (ver registro de
          experimentos). Estas tarjetas describen el riesgo esperado, no una
          señal de compra o venta.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="flex flex-col gap-4 lg:col-span-4">
          <VolatilityCard />
          <ExpectedRangeCard />
        </div>
        <div className="flex flex-col gap-4 lg:col-span-4">
          <VolatilityHistoryChart />
          <CalibrationPanel />
        </div>
        <div className="lg:col-span-4">
          <PositionSizeCalculator />
        </div>
      </div>
    </section>
  );
}
