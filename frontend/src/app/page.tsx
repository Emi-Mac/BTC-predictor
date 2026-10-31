/** Dashboard principal: layout en grid según wireframe de la Fase 5. */
import { Header } from "@/components/layout/Header";
import { CandlestickChart } from "@/components/dashboard/CandlestickChart";
import { CurrentPredictionCard } from "@/components/dashboard/CurrentPredictionCard";
import { AccuracyPanel } from "@/components/dashboard/AccuracyPanel";
import { PredictionHistoryTable } from "@/components/dashboard/PredictionHistoryTable";
import { FeatureSnapshotPanel } from "@/components/dashboard/FeatureSnapshotPanel";

export default function HomePage() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <div className="mx-auto grid w-full max-w-[1600px] flex-1 grid-cols-1 gap-4 p-4 sm:p-6 lg:grid-cols-12">
        {/* Columna principal: gráfico + historial */}
        <div className="flex flex-col gap-4 lg:col-span-8">
          <CandlestickChart />
          <PredictionHistoryTable />
        </div>

        {/* Columna lateral: predicción actual, métricas, features */}
        <div className="flex flex-col gap-4 lg:col-span-4">
          <CurrentPredictionCard />
          <AccuracyPanel />
          <FeatureSnapshotPanel />
        </div>
      </div>
    </div>
  );
}
