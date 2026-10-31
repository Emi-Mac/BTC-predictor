# 🚀 Runbook: Levantar el Frontend (Next.js Dashboard)

> **Ámbito:** `frontend/` — Dashboard de la Fase 5 (Next.js 14 + Tailwind + Lightweight Charts).
> **Audiencia:** Desarrolladores que necesitan correr el dashboard localmente (con o sin el backend real).
> **Última actualización:** 30-oct-2026.

---

## 1. Prerrequisitos

| Herramienta | Versión mínima | Verificar con |
|---|---|---|
| Node.js | ≥ 20 LTS | `node -v` |
| npm | ≥ 10 | `npm -v` |
| Backend FastAPI corriendo (opcional pero recomendado) | Fase 4 | `curl http://localhost:8000/health` |

> El frontend **puede arrancar sin el backend**, pero las cards mostrarán estados de error/skeleton
> permanentes (SWR reintentando) y el WebSocket quedará en estado `reconnecting` → `polling`.

---

## 2. Variables de entorno

Next.js **no lee el `.env` de la raíz del repo**; necesita su propio archivo dentro de `frontend/`.

```bash
cd frontend
```

Crea `frontend/.env.local`:

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

| Variable | Uso |
|---|---|
| `NEXT_PUBLIC_API_URL` | Usada por `next.config.mjs` para el rewrite `/api/* → backend` (evita CORS en dev). |
| `NEXT_PUBLIC_WS_URL` | Usada por `useLiveConnection` para abrir `${NEXT_PUBLIC_WS_URL}/ws/live`. |

> Si vas a desplegar detrás de un dominio distinto, apunta estas variables a la URL pública del backend.

---

## 3. Instalar dependencias

```bash
cd frontend
npm install
```

### ⚠️ Problema conocido: registry corporativo (401)

Si tu `npm config get registry` apunta a un Artifactory corporativo (p. ej. BBVA) y no tienes
sesión/token configurado, `npm install` fallará con:

```
npm error code E401
npm error Incorrect or missing password.
```

**Solución rápida (solo para esta instalación, no cambia tu config global):**

```bash
npm install --registry https://registry.npmjs.org/
```

**Solución permanente:** autenticarte contra el registry corporativo (`npm login` con las
credenciales de tu organización) o configurar un `.npmrc` de proyecto con el token correcto antes
de instalar.

---

## 4. Levantar el backend (dependencia opcional)

Desde la raíz del repo, en otra terminal:

```bash
docker compose --project-directory . -f infra/docker-compose.yml up -d --wait postgres
docker compose --project-directory . -f infra/docker-compose.yml run --rm --no-deps \
  -v "$PWD/backend:/app" backend alembic upgrade head
docker compose --project-directory . -f infra/docker-compose.yml up -d backend
curl http://localhost:8000/health   # debe responder 200
```

> En Windows/PowerShell reemplaza `$PWD` por `${PWD}` o la ruta absoluta entre comillas.

---

## 5. Modo desarrollo (hot reload)

```bash
cd frontend
npm run dev
```

- Abre **http://localhost:3000**
- El header debe mostrar el indicador de conexión (`🟢 En vivo` si el backend y el WS responden).
- Si ves `Cannot find module 'react'...` en el editor justo después de `npm install`, reinicia el
  servidor de TypeScript del IDE (los tipos recién instalados no siempre se recargan solos).

### Scripts disponibles

| Comando | Propósito |
|---|---|
| `npm run dev` | Servidor de desarrollo con Turbopack. |
| `npm run build` | Build de producción (`.next/`). |
| `npm run start` | Sirve el build de producción (requiere `build` previo). |
| `npm run type-check` | `tsc --noEmit`, sin emitir archivos. |
| `npm run lint:check` | ESLint en modo solo-lectura. |
| `npm run lint` | ESLint con `--fix`. |
| `npm run format:check` / `npm run format` | Prettier. |

---

## 6. Build y arranque en modo producción

```bash
cd frontend
npm run build
npm run start
# → http://localhost:3000 (puerto configurable con PORT=xxxx npm run start)
```

---

## 7. Checklist de verificación (smoke test)

1. `npm run type-check` → sin errores.
2. `npm run lint:check` → sin errores (solo puede aparecer un warning de versión de TypeScript no
   oficialmente soportada por `@typescript-eslint`, es inofensivo).
3. `npm run build` → `✓ Compiled successfully`.
4. Con el backend arriba:
   - El precio en el header carga (`$xx,xxx.xx`) y el badge `v{n} · xx.x% acc` aparece.
   - El gráfico de velas renderiza candlesticks (no skeleton infinito).
   - La card "Predicción Actual" muestra dirección + gauge de confianza + countdown corriendo.
   - El indicador de conexión pasa a `🟢 En vivo` en pocos segundos.
5. Sin backend (apágalo para probar resiliencia):
   - Las cards muestran `ErrorState` en vez de quedarse en blanco o crashear.
   - El indicador de conexión pasa a `🔄 Reconectando…` y luego `Modo polling` tras 3 intentos.

---

## 8. Troubleshooting

| Síntoma | Causa | Solución |
|---|---|---|
| `npm error code E401` al instalar | Registry corporativo sin credenciales | Ver sección 3 (usar `registry.npmjs.org` o `npm login`). |
| `Configuring Next.js via 'next.config.ts' is not supported` | Next 14.2 no soporta `next.config.ts` (solo Next ≥15) | Ya migrado a `next.config.mjs`; si reaparece, borra `next.config.ts` y usa el `.mjs`. |
| `ESLint couldn't find a configuration file` | Faltaba `.eslintrc.json` en Fase 0 | Ya añadido en `frontend/.eslintrc.json`; no debería reaparecer. |
| `error: unknown option '--turbopack'` al hacer `npm run dev` | El script usaba el flag de Next 15; Next 14.2 (instalado) usa `--turbo` | Ya corregido en `package.json` (`"dev": "next dev --turbo"`). |
| La página carga **sin ningún estilo** (fuente serif, sin dark mode, sin cards) | Faltaba `postcss.config.js` en Fase 0 → Tailwind nunca procesaba `globals.css` | Ya añadido `frontend/postcss.config.js` con los plugins `tailwindcss`/`autoprefixer`. |
| `CssSyntaxError: You cannot @apply the 'text-bull' utility here because it creates a circular dependency` (HTTP 500 en `/`) | Clases custom en `globals.css` (`.text-bull`, `.text-bear`) tenían el **mismo nombre** que las utilities de color auto-generadas por `tailwind.config.ts` (`colors.bull`, `colors.bear`), creando un `@apply` circular | Ya renombradas a `.price-up` / `.price-down` / `.price-neutral` en `globals.css`. Si agregas un nuevo color en `tailwind.config.ts`, evita crear una clase utilitaria con el mismo nombre literal. |
| El precio/predicción nunca cargan (skeleton infinito) | Backend caído o `NEXT_PUBLIC_API_URL` mal configurada | Verifica `curl $NEXT_PUBLIC_API_URL/health` y revisa la consola del navegador (`[API Error] ...`). |
| El indicador de conexión queda en `🔴 Desconectado` | WS no implementado aún en backend (`/ws/live` solo hace heartbeat) o bloqueado por proxy/firewall | Revisa Network → WS en DevTools; confirma que `ws://localhost:8000/ws/live` responde 101. |
| CORS error en consola | `CORS_ORIGINS` del backend no incluye `http://localhost:3000` | Ajusta `CORS_ORIGINS` en el `.env` del backend. |
| Puerto 3000 ocupado | Otro proceso usando el puerto | `npm run dev -- -p 3001` o libera el puerto. |
| `npm warn install-scripts ... unrs-resolver` | Script de postinstall bloqueado por npm 10+ | Informativo, no bloquea el funcionamiento; ignorar o `npm install-scripts approve unrs-resolver` si da problemas. |

---

## 9. Apagar / limpiar

```bash
# Detener el frontend: Ctrl+C en la terminal de `npm run dev` / `npm run start`

# Backend + DB (si los levantaste para probar)
docker compose --project-directory . -f infra/docker-compose.yml down

# Limpieza completa de node_modules (si hay problemas raros de dependencias)
cd frontend
rm -rf node_modules .next
npm install
```


