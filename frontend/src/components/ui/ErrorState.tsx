import { AlertTriangle } from "lucide-react";

/** Estado de error homogéneo para mostrar cuando un fetch de SWR falla. */
export function ErrorState({ message = "No se pudo cargar la información." }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-6 text-center text-sm text-gray-400">
      <AlertTriangle className="h-5 w-5 text-bear" />
      <span>{message}</span>
    </div>
  );
}

