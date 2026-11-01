/**
 * Estado de investigación del modelo direccional (registro E-01…E-12 en docs/research).
 * Se usa mientras el backend no exponga `/api/v1/metrics/model`; actualizar junto con
 * docs/research/experiment-log.md cuando cambie la evidencia.
 */
import type { ModelStatus } from "./api";

export interface ResearchStatus {
  status: ModelStatus;
  updatedAt: string;
  accuracy: number;
  baselineAccuracy: number;
  baselineName: string;
  summary: string;
  evidencePath: string;
}

export const DIRECTION_MODEL_STATUS: ResearchStatus = {
  status: "experimental",
  updatedAt: "2026-10-30",
  accuracy: 0.5402,
  baselineAccuracy: 0.5404,
  baselineName: "regla condicional a la vela previa",
  summary:
    "La predicción direccional a 4h no supera a reglas simples y no es rentable tras costos; " +
    "a 24h y 3 días tampoco supera a «siempre sube».",
  evidencePath: "docs/research/experiment-log.md",
};
