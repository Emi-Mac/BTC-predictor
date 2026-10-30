import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // ── Dark Financial Palette ──────────────────────────
        bg: {
          primary: "#0a0e17",     // Fondo principal
          surface: "#111827",     // Cards, paneles
          elevated: "#1f2937",    // Hover, inputs
        },
        brand: {
          DEFAULT: "#818cf8",     // Indigo — acentos de marca
          dark: "#4f46e5",
        },
        // Precios alcistas/bajistas (estándar financiero)
        bull: {
          DEFAULT: "#4ade80",     // Verde alcista
          dim: "#166534",         // Verde atenuado (background)
        },
        bear: {
          DEFAULT: "#f87171",     // Rojo bajista
          dim: "#7f1d1d",         // Rojo atenuado
        },
        neutral: {
          DEFAULT: "#facc15",     // Amarillo — confianza media
          dim: "#713f12",
        },
        border: {
          DEFAULT: "#374151",
          subtle: "#1f2937",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.3s ease-in-out",
        "slide-up": "slideUp 0.3s ease-out",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { transform: "translateY(8px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};

export default config;

