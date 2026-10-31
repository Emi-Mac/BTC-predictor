import { DIRECTION_MODEL_STATUS } from "@/lib/modelStatus";

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

// Placeholder de la página principal — el dashboard se completará en Fase 5
export default function HomePage() {
  const model = DIRECTION_MODEL_STATUS;
  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="max-w-xl text-center">
        <h1 className="mb-2 text-3xl font-bold text-brand">
          BTC Direction Predictor
        </h1>
        <p className="text-gray-400">Dashboard en construcción — Fase 5</p>
        <div className="mt-4 inline-flex items-center gap-2 rounded-full bg-bg-surface px-4 py-2 text-sm text-gray-300">
          <span className="h-2 w-2 animate-pulse rounded-full bg-bull"></span>
          Sistema operativo
        </div>
        <section
          aria-label="Estado del modelo direccional"
          className="mt-6 rounded-lg border border-yellow-700/60 bg-bg-surface p-4 text-left text-sm text-gray-300"
        >
          <p className="mb-2 font-semibold uppercase tracking-wide text-yellow-300">
            Modelo direccional:{" "}
            {model.status === "experimental" ? "experimental" : "validado"}
          </p>
          <p>{model.summary}</p>
          <p className="mt-2 font-mono text-gray-400">
            Acierto {percent(model.accuracy)} vs{" "}
            {percent(model.baselineAccuracy)} ({model.baselineName})
          </p>
          <p className="mt-2 text-xs text-gray-500">
            Evidencia: {model.evidencePath} · actualizado {model.updatedAt}
          </p>
        </section>
      </div>
    </div>
  );
}
