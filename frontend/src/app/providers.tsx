"use client";

/** Providers globales: SWRConfig (fetcher + polling fallback), WebSocket y Toaster. */
import type { ReactNode } from "react";
import { SWRConfig } from "swr";
import { fetcher } from "@/lib/api";
import { useLiveConnection } from "@/hooks/useLiveConnection";
import { Toaster } from "@/components/ui/Toaster";

function LiveConnectionMount() {
  useLiveConnection();
  return null;
}

export function Providers({ children }: { children: ReactNode }) {
  return (
    <SWRConfig
      value={{
        fetcher,
        revalidateOnFocus: true,
        dedupingInterval: 5_000,
        errorRetryCount: 3,
      }}
    >
      <LiveConnectionMount />
      {children}
      <Toaster />
    </SWRConfig>
  );
}

