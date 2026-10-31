import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/utils";

interface CardProps extends Omit<HTMLAttributes<HTMLDivElement>, "title"> {
  title?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
}

/** Card base del dashboard, con título opcional y slot de acción a la derecha. */
export function Card({ title, action, children, className, ...rest }: CardProps) {
  return (
    <div className={cn("card flex flex-col gap-3 animate-fade-in", className)} {...rest}>
      {(title || action) && (
        <div className="flex items-center justify-between">
          {title && (
            <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-400">
              {title}
            </h2>
          )}
          {action}
        </div>
      )}
      {children}
    </div>
  );
}


