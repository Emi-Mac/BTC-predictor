"use client";

import { Wifi, WifiOff, RefreshCw, RadioTower } from "lucide-react";
import { useConnectionStore } from "@/store/connectionStore";
import { cn } from "@/lib/utils";

const CONFIG = {
  open: { icon: Wifi, label: "En vivo", className: "text-bull" },
  connecting: { icon: RadioTower, label: "Conectando…", className: "text-neutral" },
  reconnecting: { icon: RefreshCw, label: "Reconectando…", className: "text-neutral" },
  polling: { icon: RefreshCw, label: "Modo polling", className: "text-neutral" },
  closed: { icon: WifiOff, label: "Desconectado", className: "text-bear" },
} as const;

/** Indicador visual del estado del WebSocket (header). */
export function ConnectionStatus() {
  const status = useConnectionStore((s) => s.status);
  const { icon: Icon, label, className } = CONFIG[status];

  return (
    <div
      className={cn("flex items-center gap-1.5 text-xs font-medium", className)}
      title={label}
      aria-label={`Estado de conexión: ${label}`}
    >
      <Icon className={cn("h-3.5 w-3.5", status === "reconnecting" && "animate-spin")} />
      <span className="hidden sm:inline">{label}</span>
    </div>
  );
}

