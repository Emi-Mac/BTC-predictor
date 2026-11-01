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

### Pronósticos de volatilidad en vivo (F3)

Con `VOLATILITY_SOURCES_ENABLED=true`, la migración 009 aplicada y modelos **activos**, cada ciclo
del ML worker (`run_features`) hace dos cosas, cada una aislada de las fallas de la otra y de
la predicción direccional:

1. **Resolver** los pronósticos cuyo horizonte ya se cumplió: varianza realizada = media de los
   cuadrados de los retornos log de las velas `t+1…t+h` (la definición del objetivo de
   entrenamiento), retorno `log(c[t+h]/c[t])`, cobertura de las bandas y QLIKE. Solo se resuelve
   con las `h` velas siguientes sin huecos y ya cerradas.
2. **Pronosticar** la última vela cerrada a 6 y 42 velas, de forma idempotente por
   (vela, horizonte). Se sirve el primario; si DVOL, funding, features de la vela, historial o el
   calendario FOMC no cumplen su regla de frescura se sirve el respaldo HAR y se guardan
   `fallback_reason` y `features_as_of` (marca de tiempo de cada fuente usada).

Las bandas usan `k68/k90/k95` conformes: la semilla del artefacto más los pronósticos en vivo del
mismo modelo ya resueltos (ventana de 2 años, mínimo 1,080). Si no hay suficientes, los
multiplicadores quedan en `NULL` y la UI no debe mostrar bandas. Sin un primario activo para un
horizonte no se emite pronóstico para ese horizonte.

```bash
docker exec btc_postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c \
  "select horizon_candles, model_version, fallback_reason, forecast_vol_annual, regime, \
          k95, resolved_at is not null as resolved from volatility_forecasts order by id desc limit 6"
```

### API de volatilidad (F4)

Con `VOLATILITY_ENABLED=true` en el backend (y las migraciones 007–009 aplicadas) se exponen
estos endpoints; con `false` (por defecto) responden 404 y no se programa la difusión WS, de modo
que el backend puede convivir con una base sin esas tablas.

| Método | Ruta | Contenido |
|---|---|---|
| GET | `/api/v1/volatility/current?horizon=1d\|7d` | Volatilidad anual y del horizonte, régimen, bandas 68/90/95% (`null` sin calibración suficiente), DVOL, realizada 7d/30d, `model_version`, `fallback`, `validation_status` |
| GET | `/api/v1/volatility/history?horizon=&limit=&offset=` | Pronóstico vs realizado por vela, paginado |
| GET | `/api/v1/metrics/volatility?horizon=` | Resueltos, cobertura 68/90/95%, QLIKE y QLIKE del ingenuo, pendiente de Mincer-Zarnowitz (≥ 30 resueltos) y cobertura de las últimas 12 semanas |
| GET | `/api/v1/risk/position-size?capital=&risk_pct=&stop_k=&horizon=&side=` | Tamaño de posición para arriesgar `risk_pct`% hasta un stop a `stop_k`·σ; rechaza `risk_pct > 5`, capital ≤ 0 y `stop_k` fuera de (0, 10] con 422 |
| WS | `/ws/live` | `{ "type": "volatility", "data": { horizon, vol_annual, regime, bands } }` una vez por pronóstico nuevo |

- `validation_status` sigue en `in_validation` (`VOLATILITY_VALIDATION_STATUS`) hasta completar la
  sombra prospectiva de 12 semanas; la calculadora lo advierte en `warnings`.
- `/api/v1/metrics/model` y `/api/v1/predictions/*` añaden `status`
  (`DIRECTION_MODEL_STATUS`, «experimental» por defecto), `beats_baseline`,
  `best_baseline_accuracy` y `evidence_url`; `/api/v1/metrics/accuracy` añade `baseline_overall`
  (la regla «siempre la dirección más frecuente» sobre las mismas predicciones resueltas).
- La volatilidad no es una señal direccional: ninguna respuesta indica comprar o vender.

### Sombra prospectiva de 12 semanas (F6)

Stack aislado (`infra/docker-compose.shadow.yml`, proyecto `btc-shadow`): base, red, volúmenes y
puertos propios (PostgreSQL en 5433, API de inspección en 18100) y una imagen **inmutable** del
worker (el código se copia, no se monta), de modo que lo evaluado no cambia durante las 12
semanas. Operación con `infra/scripts/shadow.sh`:

```bash
infra/scripts/shadow.sh up       # imágenes, migraciones 001–009, entrena y activa 1d/7d, congela el protocolo, levanta el worker
infra/scripts/shadow.sh status   # contenedores, modelos activos, pronósticos y salud de ingesta
infra/scripts/shadow.sh report   # evalúa los criterios (guarda ml/artifacts/shadow/report-*.json)
infra/scripts/shadow.sh api      # API de inspección en http://localhost:18100/docs
infra/scripts/shadow.sh down     # detiene conservando la base;  destroy: la borra
```

`up` **inicia el reloj**: `ml/artifacts/shadow/manifest.json` congela criterios, modelos activos
(con su SHA-256) y el commit, empieza a la siguiente medianoche UTC y no se reescribe. El primer
arranque tarda unos minutos (backfill de 1,200 días de velas y de DVOL, OI y FOMC). Durante la
sombra no se reentrena ni se cambia de modelo (`models_frozen` lo vigila) y la UI no la muestra.

**Criterios** (`python -m src.volatility.shadow report`, ver el docstring del módulo):

| Criterio | Regla |
|---|---|
| 1 día | 1 observación por día (vela de 00:00 UTC, ≈ 84): el IC binomial exacto del 95% de la cobertura de 68% y 90% debe contener el nivel nominal; muestra ≥ 90% de lo esperado |
| 7 días | Solo monitoreo (vela de 00:00 UTC de los lunes, ≈ 12); sigue «en validación» hasta 26 semanas |
| Ingesta | DVOL y OI registrados con ≤ 6 h de rezago y ningún hueco > 6 h entre pronósticos |
| Respaldo | < 5% de los pronósticos servidos con HAR |

Hasta cumplir las 12 semanas el informe es provisional (`in_progress`; `interim`: `on_track` o
`failing`). Con 84 observaciones el IC es de ±10 pp: es una prueba de cordura, no de precisión.
`recommended_validation_status` solo pasa a `validated` con la sombra aprobada **y** 26 semanas
(entonces se cambia `VOLATILITY_VALIDATION_STATUS`).

Este mismo stack sirve para la observación de 48 h de F1 (datos en vivo ≤ 3 h): `shadow.sh status`
ejecuta `ingestion.run check`, que incluye DVOL, OI y calendario FOMC.
