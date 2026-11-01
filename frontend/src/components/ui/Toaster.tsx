"use client";

/**
 * Contenedor de notificaciones toast (esquina inferior derecha).
 * Auto-descarta cada toast a los 5s.
 */
import { useEffect } from "react";
import { CheckCircle2, Info, AlertTriangle, XCircle } from "lucide-react";
import { useToastStore, type Toast, type ToastVariant } from "@/store/toastStore";
import { cn } from "@/lib/utils";

const ICONS: Record<ToastVariant, typeof Info> = {
  info: Info,
  success: CheckCircle2,
  warning: AlertTriangle,
  error: XCircle,
};

const VARIANT_CLASSES: Record<ToastVariant, string> = {
  info: "border-brand/40 text-brand",
  success: "border-bull/40 text-bull",
  warning: "border-neutral/40 text-neutral",
  error: "border-bear/40 text-bear",
};

function ToastItem({ toast }: { toast: Toast }) {
  const removeToast = useToastStore((s) => s.removeToast);
  const Icon = ICONS[toast.variant];

  useEffect(() => {
    const timer = setTimeout(() => removeToast(toast.id), 5000);
    return () => clearTimeout(timer);
  }, [toast.id, removeToast]);

  return (
    <div
      className={cn(
        "animate-slide-up flex items-center gap-2 rounded-lg border bg-bg-surface px-4 py-2.5 text-sm shadow-lg",
        VARIANT_CLASSES[toast.variant],
      )}
    >
      <Icon className="h-4 w-4 shrink-0" />
      <span className="text-gray-200">{toast.message}</span>
    </div>
  );
}

export function Toaster() {
  const toasts = useToastStore((s) => s.toasts);

  return (
    <div className="fixed bottom-4 right-4 z-50 flex w-full max-w-sm flex-col gap-2">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} />
      ))}
    </div>
  );
}

