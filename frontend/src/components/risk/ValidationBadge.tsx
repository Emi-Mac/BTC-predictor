import type { ValidationStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

/** Etiqueta del estado de la evidencia: «en validación» hasta completar la sombra de 12 semanas. */
export function ValidationBadge({ status }: { status: ValidationStatus }) {
  const validated = status === "validated";
  return (
    <span
      title={
        validated
          ? "Superó la validación prospectiva."
          : "Sombra prospectiva en curso (12 semanas): la cobertura real aún no está confirmada."
      }
      className={cn(
        "rounded-full border px-2 py-0.5 text-[11px]",
        validated
          ? "border-sky-400/50 bg-sky-400/10 text-sky-300"
          : "border-amber-400/50 bg-amber-400/10 text-amber-300",
      )}
    >
      {validated ? "Validado" : "En validación"}
    </span>
  );
}
