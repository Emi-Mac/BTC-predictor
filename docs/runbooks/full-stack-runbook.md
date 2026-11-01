# 🚀 Runbook: stack completo (base de datos + ML worker + backend + frontend)

> **Ámbito:** levantar y operar todo el plan original en local. Para el detalle del frontend
> (scripts, troubleshooting) ver [frontend-runbook.md](./frontend-runbook.md).
> **Última actualización:** 31-oct-2026 (verificado de extremo a extremo en la rama
> `feature/fase4-desarrollo-API`).

## 1. Cómo fluyen los datos

```text
Binance / yfinance / Alternative.me
        │  cada 4h (cron + watchdog)
        ▼
 ml-worker ──► ohlcv_candles ──► computed_features + contextual_features
                                          │
                                          ▼
 backend (scheduler de predicción, sondeo cada 5 min, idempotente)
   · lee el modelo activo de model_registry (ONNX)
   · arma el vector de features en el orden del modelo
   · guarda en predictions_log y resuelve la predicción anterior
        │  REST /api/v1/*  ·  WebSocket /ws/live
        ▼
 frontend (Next.js, http://localhost:3000)
```

- `computed_features` guarda los indicadores base. El **contexto de precio** (`return_*`,
  `candle_body`, `relative_macd`, …) que necesita el modelo activo vive en
  `contextual_features` (JSONB por vela, migración `006`). El ML worker lo recalcula en cada
  ciclo con `ml/src/features/context_store.py`.
- El backend **no** entrena ni calcula features: solo sirve el modelo activo.

## 2. Levantar todo

Desde la raíz del repo (la base de datos conserva sus datos entre reinicios):

```bash
cp .env.example .env            # solo la primera vez; ajusta POSTGRES_PASSWORD y SECRET_KEY
C="docker compose --project-directory . -f infra/docker-compose.yml"

$C up -d --wait postgres
$C build backend ml-worker
$C run --rm --no-deps -v "$PWD/backend:/app" backend alembic upgrade head   # incluye la 006
$C up -d backend ml-worker

# Solo la primera vez, o si cambia el código de features: contexto de todo el histórico
docker exec btc_ml_worker python -m src.features.context_store

# Frontend (otra terminal)
cd frontend
printf 'NEXT_PUBLIC_API_URL=http://localhost:8000\nNEXT_PUBLIC_WS_URL=ws://localhost:8000\n' > .env.local
npm ci --registry https://registry.npmjs.org/ && npm run dev
```

Abrir **http://localhost:3000**. API y documentación: http://localhost:8000/docs.

> Los contenedores montan `./backend/src` y `./ml/artifacts` **del directorio desde el que se
> ejecuta compose**. Si usas un `git worktree`, ejecútalo desde él y enlaza `ml/artifacts` a la
> carpeta real de artefactos; si no, el backend no encontrará el modelo ONNX.

## 3. Modelo activo

El entrenamiento **nunca** activa un modelo (`ml/src/training/registry.py`). Se activa a
propósito con:

```bash
docker exec btc_ml_worker python -m src.training.activation --version 2 --dry-run
docker exec btc_ml_worker python -m src.training.activation --version 2 \
  --reason "motivo de la activación"
docker restart btc_backend      # el scheduler emite la predicción pendiente al arrancar
```

La activación valida que el backend pueda servir el modelo (archivos del artefacto,
`feature_names` coherentes, features persistidas y familia soportada; el LSTM no lo está),
deja **una sola** versión activa y anota fecha y motivo en `model_registry.notes`.

| Versión | Familia | Accuracy en su evaluación | Baseline | Estado |
|---|---|---|---|---|
| 1 | LSTM | 53.3% (último fold) | 53.3% | Inactivo (no servible por el backend) |
| 2 | Regresión logística | 55.4% (holdout oct-2026, 168 velas; IC 95% 47.8–62.7%) | 48.2% | **Activo** |

La v2 es la de mayor accuracy medida de forma independiente, **pero no cumple los criterios de
calidad del plan** (+3 pp sobre el mejor baseline en desarrollo; margen real +0.99 pp). Es una
decisión de producto mantenerla activa; la evidencia está en la rama
`feature/prediccion-volatilidad` (`docs/research/`).

## 4. Verificación rápida

```bash
curl -s localhost:8000/readiness
curl -s localhost:8000/api/v1/predictions/current
curl -s localhost:8000/api/v1/metrics/model | head -c 300
curl -s localhost:8000/api/v1/metrics/accuracy
docker exec btc_postgres sh -c 'psql -U $POSTGRES_USER -d $POSTGRES_DB -c \
  "select * from model_registry order by version"'
```

Comportamiento esperado: una predicción nueva cada 4h, unos 5–90 minutos después del cierre de
la vela (depende de cuándo el ML worker ingesta y calcula features), y la anterior queda
resuelta. En el log del ML worker aparece `ingestion_recovery_needed` cuando la vela todavía no
estaba disponible a la hora del cron: es el watchdog reintentando y es normal.

## 5. Pruebas

```bash
# Backend (requiere una base de pruebas separada para los tests de integración)
docker exec btc_postgres sh -c 'psql -U $POSTGRES_USER -d $POSTGRES_DB \
  -c "create database btcpredictor_test"'
# imagen de pruebas: backend + pytest/ruff/mypy
docker build --target development -t btc-predictor-backend:test backend
printf 'FROM btc-predictor-backend:test\nRUN pip install pytest pytest-asyncio pytest-cov ruff mypy\n' \
  | docker build -t btc-backend-test -
docker run --rm --network btc-predictor_btc_network -v "$PWD/backend:/app" -w /app \
  -e TEST_DATABASE_URL="postgresql+asyncpg://USER:PASS@postgres:5432/btcpredictor_test" \
  btc-backend-test python -m pytest -q

# Frontend
cd frontend && npx tsc --noEmit && npm run lint:check && npm run build
```

Los tests de integración crean un esquema temporal por prueba y lo eliminan; no tocan las
tablas reales. El ML se prueba igual (ver `docs/research/methodology.md` en la rama de
volatilidad, sección «Cómo ejecutar»).

## 6. Problemas conocidos y notas

- **Muestra pequeña:** el panel de métricas calcula el accuracy con las predicciones resueltas
  desde la activación (6 al 1-nov-2026: 50.0%). Con tan pocas no es informativo.
- **Lint del backend:** `ruff` reporta ~80 avisos preexistentes (imports de solo tipos, `S104`,
  `S105`) y `mypy` ~14 errores de tipado; no afectan a la ejecución. No se corrigieron en esta
  tarea.
- **Prettier:** varios archivos del frontend ya estaban sin formatear antes de esta rama.
- **Corregidos al verificar la rama:** el gráfico de velas no se creaba (el contenedor solo
  existía tras la carga) y el contador «Próximo cierre» quedaba en `0h 00m 00s` (la predicción
  de la vela `T` se refiere a la siguiente, que cierra en `T + 8h`).

## 7. Apagar

```bash
docker compose --project-directory . -f infra/docker-compose.yml stop backend ml-worker
# Ctrl+C en `npm run dev`
```

## 8. Fuentes de volatilidad (rama `feature/prediccion-volatilidad`)

La migración **007** crea `implied_volatility` (DVOL de Deribit, velas horarias cerradas),
`open_interest` (Binance, 5 min) y `event_calendar` (reuniones del FOMC). El ML worker las
ingesta tras las velas, con `VOLATILITY_SOURCES_ENABLED=true`; apagado por defecto.

> **No aplicar la 007 a una base compartida con la rama de la Fase 4:** su Alembic no conoce esa
> revisión y dejaría de arrancar. Usa una base y un stack propios para esta rama.

```bash
$C run --rm --no-deps -v "$PWD/backend:/app" backend alembic upgrade head   # llega a la 007
# en .env: VOLATILITY_SOURCES_ENABLED=true; luego reiniciar el ml-worker
docker exec btc_ml_worker python -m src.ingestion.run check   # incluye DVOL, OI y FOMC
```

- **DVOL:** solo se guardan velas ya cerradas (la hora en curso aún cambia). Se re-piden las
  últimas 24 h en cada pasada; el primer arranque rellena desde 7 días antes de la primera vela.
- **OI:** Binance solo conserva 30 días; el primer arranque trae 29. Con `startTime` y `endTime`
  devuelve las **últimas** 500 filas del rango, por lo que el cliente pagina en ventanas.
- **FOMC:** se refresca cada `FOMC_REFRESH_DAYS` (7) o en cuanto no queda una reunión futura.
- **Salud:** `check` y el watchdog marcan DVOL u OI con más de `DVOL_MAX_LAG_HOURS` /
  `OI_MAX_LAG_HOURS` (3 h) de rezago, o un calendario sin reuniones futuras.
- Las filas son inmutables (`ON CONFLICT DO NOTHING`); `recorded_at` / `captured_at` indican
  cuándo se leyó el dato, no cuándo estuvo disponible.

### Modelos de volatilidad (F2)

```bash
# Evaluación E-12b: aceptación, calibración de bandas y paridad con producción
docker exec btc_ml_worker python -m src.volatility.study \
  --dataset artifacts/datasets/btcusdt-20200901-20261001-v1 \
  --aux artifacts/aux-sources/aux-20190901-20261001-v2 --development-end 2026-10-01T00:00:00+00:00

# Entrena primario (har_x_live) y respaldo (har) de un horizonte y los registra INACTIVOS (migración 008)
docker exec btc_ml_worker python -m src.volatility.train --horizon 42 --register \
  --dataset ... --aux ... --development-end ...        # --horizon 6 para 1 día

# Activación explícita: solo cambia la versión activa del mismo horizonte y rol
docker exec btc_ml_worker python -m src.volatility.activation --version 3 --dry-run
docker exec btc_ml_worker python -m src.volatility.activation --version 3 --reason "E-12b aceptado"
```

- Los artefactos son JSON autocontenidos (coeficientes, varianza residual, umbrales de régimen,
  métricas de validación y la semilla de calibración); la activación verifica su huella SHA-256.
- A 1 día E-12b no demuestra ventaja de `har_x_live` sobre HAR: decide con esa evidencia a
  la vista cuál activar como primario de 1d.
- Por ahora el entrenamiento usa el dataset congelado; el reentrenamiento mensual desde
  PostgreSQL llega con la inferencia en el worker (F3) reutilizando `live_inputs`.
