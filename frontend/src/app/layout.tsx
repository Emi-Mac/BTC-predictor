import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "BTC Direction Predictor",
  description:
    "Plataforma experimental de predicción direccional de Bitcoin. ⚠️ No es asesoría financiera.",
  keywords: ["Bitcoin", "BTC", "Machine Learning", "Predicción", "Crypto"],
  robots: "noindex, nofollow", // No indexar en buscadores (herramienta interna)
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es" className="dark">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} bg-bg-primary font-sans text-gray-100 antialiased`}
      >
        {/* Banner de disclaimer — siempre visible */}
        <div className="sticky top-0 z-50 w-full bg-yellow-900/80 px-4 py-1.5 text-center text-xs text-yellow-200 backdrop-blur-sm">
          ⚠️ <strong>DISCLAIMER:</strong> Esta herramienta es únicamente para fines educativos e
          investigación. <strong>NO constituye asesoría financiera</strong>. Las predicciones del
          modelo son experimentales y pueden ser incorrectas.
        </div>

        <Providers>
          <main className="min-h-screen">{children}</main>
        </Providers>
      </body>
    </html>
  );
}