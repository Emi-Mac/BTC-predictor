# BTC Direction Predictor 🔮

> ⚠️ **DISCLAIMER**: Esta es una herramienta experimental de investigación y aprendizaje. **NO constituye asesoría financiera**. Las predicciones del modelo ML son experimentales y pueden ser incorrectas con frecuencia.

Plataforma Full-Stack para predicción direccional binaria del precio de Bitcoin (UP/DOWN) con un objetivo realista de accuracy del 55–60%.

---

## Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| ML Pipeline | Python 3.12, XGBoost, PyTorch, ONNX |
| Backend API | FastAPI (Arquitectura Hexagonal) |
| Base de Datos | PostgreSQL 16 |
| Frontend | Next.js 14, Tailwind CSS, TradingView Lightweight Charts |
| Contenedores | Docker + Docker Compose |

## Fuentes de Datos (todas gratuitas)

- **Binance Public API** — OHLCV histórico y en tiempo real (sin autenticación)
- **yfinance** — DXY, S&P 500, VIX, Gold (datos macroeconómicos)
- **Alternative.me** — Crypto Fear & Greed Index

---

## Inicio Rápido

### Prerrequisitos

- Docker ≥ 24.0
- Docker Compose ≥ 2.0
- Python ≥ 3.12 (para desarrollo local del ML)
- Node.js ≥ 20 (para desarrollo local del frontend)

### 1. Clonar y configurar entorno

```bash
git clone <repo-url> btc-predictor
cd btc-predictor

# Copiar variables de entorno
cp .env.example .env

# Editar .env con tus valores (especialmente POSTGRES_PASSWORD y SECRET_KEY)
```

### 2. Levantar PostgreSQL

```bash
docker compose --project-directory . -f infra/docker-compose.yml up -d --wait postgres
```

### 3. Aplicar migraciones de base de datos

```bash
# Ejecutar Alembic desde la raíz, sin instalar dependencias en la máquina
docker compose --project-directory . -f infra/docker-compose.yml run --rm --no-deps \
  -v "$PWD/backend:/app" backend alembic upgrade head
```

> Aplica las migraciones **antes** de levantar el backend. Docker desactiva la creación automática
> de tablas (`BACKEND_AUTO_CREATE_SCHEMA=false`) para que Alembic mantenga la versión del esquema.
> Fuera de Docker, esa opción conserva su valor anterior por defecto (`true`).

### 3b. Backend, ingesta y features

```bash
# Backfill inicial + cálculo de features + actualización automática cada 4h
docker compose --project-directory . -f infra/docker-compose.yml up -d backend ml-worker

# Verificar integridad del dataset (exit 0 = OK)
docker exec btc_ml_worker python -m src.ingestion.run check

# Recalcular features offline sin volver a descargar las fuentes
docker exec btc_ml_worker python -m src.features.run compute

# Selección usando únicamente etiquetas conocidas hasta la fecha de entrenamiento (UTC)
docker exec btc_ml_worker python -m src.features.run select --train-end 2026-01-01T00:00:00+00:00

# Una sola vez tras aplicar la migración 004: recuperar volúmenes agresores históricos
docker exec btc_ml_worker python -m src.ingestion.backfill_taker

# Sincronizar funding histórico o reintentar si la fuente estuvo indisponible
docker exec btc_ml_worker python -m src.ingestion.funding_sync
```

Las features se actualizan después de cada ingesta y se guardan mediante upsert en
`computed_features`. Las filas sin ventanas completas o sin datos externos vigentes se excluyen;
no se rellenan con información futura. La vela cerrada más reciente se conserva para inferencia
con `target_label = NULL`; se excluye del entrenamiento hasta conocer el siguiente cierre.
El corte de selección debe fijarse por cada fold de entrenamiento y la selección se debe repetir
en cada fold, nunca ajustarse usando validación o test.

Los cierres macro del día D se usan desde D+1 a las 00:00 UTC, con arrastre limitado a cinco días.
Los cambios macro son entre cierres observados; la correlación BTC–S&P usa 20 rendimientos de
fechas bursátiles emparejadas. FGI SMA y momentum usan 7 días calendario. VWAP se reinicia a
medianoche UTC; la volatilidad se anualiza con 365 días. Los porcentajes y tasas se expresan como
fracciones (0.01 = 1 %). Hora y día de semana corresponden al cierre de la vela en UTC.

Las velas guardan también el volumen de compra agresora de Binance en BTC y en USDT. Las nuevas
velas lo incorporan automáticamente. El backfill es idempotente: solo completa valores ausentes
y no modifica los campos OHLCV originales; aborta la transacción si Binance devuelve volúmenes
totales distintos o faltan velas. Los datos no disponibles en velas sintéticas permanecen nulos.
Los ratios y el desequilibrio se calculan con velas cerradas; si el volumen total es cero, el
ratio queda nulo y cualquier imputación se ajusta únicamente en entrenamiento.

El worker revisa cada 60 segundos si faltan velas recientes, si las features están atrasadas o si
el funding supera la tolerancia configurada. Tras una suspensión o desconexión intenta recuperar
los datos y reprograma el cron; no puede ejecutarse mientras Docker o el equipo están detenidos.
`INGESTION_WATCHDOG_SECONDS` controla la frecuencia. El estado de salud de Docker usa
`python -m src.ingestion.run check`, que ahora informa la última vela esperada y falla si los
datos están atrasados. El backend sigue exponiendo su propio estado de salud de aplicación.

El funding de futuros de Binance se guarda por símbolo y hora del evento, sin sobrescribir eventos
existentes. Para las features se considera disponible **una vela después** del timestamp de
funding y se deja de propagar tras 36 horas; la vigencia se puede ajustar con
`FUNDING_FEATURE_MAX_AGE_HOURS`. Esta demora es una hipótesis conservadora: el endpoint histórico
no demuestra la hora exacta de publicación ni conserva necesariamente versiones anteriores.
`FUNDING_ENABLED=false` desactiva esta fuente; con ella activa, el health check exige un evento
reciente dentro de `FUNDING_MAX_LAG_HOURS` (24 por defecto).

La consulta de interés abierto reciente de Binance funcionó, pero el endpoint rechazó solicitudes
del inicio del histórico de 2023. No se fabrican valores para cubrir los años faltantes ni se
utiliza interés abierto en el modelo histórico.

### 3c. Entrenamiento y evaluación (Fase 3)

```bash
# Construir la imagen actualizada y ejecutar un experimento offline
docker compose --project-directory . -f infra/docker-compose.yml build ml-worker
docker compose --project-directory . -f infra/docker-compose.yml run --rm --no-deps \
  ml-worker python -m src.training.run
```

El experimento compara XGBoost y LSTM con tres baselines en diez meses completos de validación
walk-forward. En cada fold se reservan meses anteriores separados para buscar hiperparámetros y
ajustar la calibración Platt. Imputación, selección y escalado se ajustan solo con el train de
cada división; las etiquetas que aún no se conocen al comienzo de la siguiente ventana se
excluyen. Los primeros nueve folds eligen la familia y el décimo se reserva como evaluación final.

Los resultados y las curvas de calibración por intervalos se guardan en
`ml/artifacts/run-*/walk_forward.json`, junto con `model.onnx`, `preprocessor.json` y
`calibration.json`. La ruta de inferencia [OnnxPredictor](./ml/src/export/onnx_model.py) aplica
preprocesamiento y calibración: XGBoost, LightGBM y regresión logística reciben `[batch, features]` y el LSTM
`[batch, 20, features]`, en el orden de `preprocessor.json`. El ONNX corresponde al modelo
evaluado en el último fold y no se reentrena con ese fold ni con datos posteriores.

El modelo se registra siempre **inactivo**. El comando devuelve código 2 si no supera tanto el
mejor baseline de cada fold por más de 3 puntos porcentuales en promedio, como el mejor baseline
del último fold por más de 2 puntos, o si falla algún otro criterio registrado. Un resultado
por debajo de esos umbrales no es un modelo apto para promover.

#### Investigación para mejorar la precisión

El comando de investigación compara variantes predefinidas con XGBoost y regresión logística,
ventanas expansivas o de 540 días, señales de precio causales y calibración Platt o identidad.
Las opciones `--include-taker`, `--include-lightgbm` y `--include-funding` añaden,
respectivamente, las señales agresoras, los candidatos LightGBM y las features de funding; están
desactivadas por defecto.
Selecciona una sola variante usando diez folds anteriores al período reservado; el holdout no
participa en la selección. Los límites deben corresponder a velas ya cerradas con etiquetas
conocidas, y el período no debe haberse utilizado antes como holdout:

```bash
# Sustituir por un período realmente nuevo y mantener ambos límites en UTC
HOLDOUT_START="AAAA-MM-DDT00:00:00+00:00"
HOLDOUT_END="AAAA-MM-DDT00:00:00+00:00"
docker compose --project-directory . -f infra/docker-compose.yml run --rm --no-deps \
  ml-worker python -m src.training.research develop \
  --holdout-start "$HOLDOUT_START" --holdout-end "$HOLDOUT_END"

# Usar el directorio /app/artifacts/research-... devuelto por el comando anterior
RESEARCH_DIR="/app/artifacts/research-..."
docker compose --project-directory . -f infra/docker-compose.yml run --rm --no-deps \
  ml-worker python -m src.training.research evaluate --directory "$RESEARCH_DIR"
```

`development.json` conserva el ranking y la huella del dataset; `holdout.json` contiene el
resultado de la variante seleccionada. El archivo `holdout.lock` evita puntuarla otra vez y se
rechazan períodos que se solapen con un holdout ya evaluado. El código de salida 2 indica que no
se cumplieron todos los criterios de calidad, aunque el ONNX quede guardado e inactivo.

Las señales adicionales se calculan **en lote** desde las velas y features persistidas mediante
[load_contextual_features](./ml/src/features/context_repository.py); no se añaden todavía a
`computed_features`. Antes de inferir con un modelo que las requiera, hay que construir ese
contexto y pasar al ONNX las columnas indicadas en `preprocessor.json`. Para incluir flujo y
funding se usa `load_contextual_features(engine, settings, include_taker=True, include_funding=True)`.

Para una comparación exploratoria con y sin flujo agresor y con LightGBM, sin puntuar un holdout:

```bash
docker exec btc_ml_worker python -m src.training.taker_ablation \
  --development-end 2026-10-01T00:00:00+00:00 --include-lightgbm --include-funding
```

Ese corte usa folds de desarrollo ya observados y no constituye una prueba independiente. El
informe se guarda en `ml/artifacts/taker-ablation-*/ablation.json`; cualquier variante elegida
requiere un holdout posterior que todavía no se haya usado antes de considerar su promoción.
La ablación de octubre de 2026 mostró 53,37 % con funding frente a 53,15 % sin él en desarrollo;
esa diferencia es exploratoria y no cumple el margen de tres puntos sobre el baseline.

### 4. Desarrollar el frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

### 5. Instalar pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

---

## Estructura del Proyecto

```
btc-predictor/
├── ml/                    # Pipeline de Machine Learning
│   ├── src/
│   │   ├── ingestion/     # Conectores a APIs externas
│   │   ├── features/      # Feature engineering
│   │   ├── training/      # Walk-Forward training loop
│   │   ├── evaluation/    # Métricas y comparación vs baseline
│   │   └── export/        # Exportación ONNX
│   └── notebooks/         # EDA y prototipado
├── backend/               # API FastAPI (Arquitectura Hexagonal)
│   ├── src/
│   │   ├── domain/        # Entidades y puertos (interfaces)
│   │   ├── application/   # Casos de uso / servicios
│   │   ├── adapters/      # Implementaciones concretas
│   │   └── config/        # Settings, DB, DI
│   └── migrations/        # Alembic migrations
├── frontend/              # Next.js + Tailwind CSS
│   └── src/
│       ├── app/           # Next.js App Router
│       ├── components/    # Componentes UI
│       ├── hooks/         # Custom hooks
│       └── lib/           # API client, utilidades
├── infra/                 # Docker Compose y scripts
└── .env.example           # Template de variables de entorno
```

---

## Plan de Fases

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0 — Bootstrap | Monorepo, Docker, DB, scaffolding | ✅ Completada |
| 1 — Ingesta | Conectores Binance, yfinance, FGI | ✅ Completada |
| 2 — Features | Indicadores técnicos y macroeconómicos | ✅ Completada |
| 3 — Modelado | Walk-Forward, XGBoost, LSTM, ONNX | 🟡 Implementada; rendimiento pendiente |
| 4 — API | FastAPI hexagonal + endpoints REST | ✅ Completada (80 tests, 88,5% coverage); benchmark de latencia de inferencia pendiente |
| 5 — Frontend | Dashboard dark-mode + TradingView | 🔲 Pendiente |
| 6 — Integración | E2E tests, hardening, model drift | 🔲 Pendiente |

---

## Comandos Útiles

```bash
# Backend: iniciar en modo desarrollo
cd backend && uvicorn src.main:app --reload

# Backend: generar nueva migración
cd backend && alembic revision --autogenerate -m "descripción"

# Backend: ver migraciones pendientes
cd backend && alembic history

# ML: ingesta manual (backfill|update|check|serve); requiere DATABASE_URL y `alembic upgrade head`
cd ml && python -m src.ingestion.run update

# Frontend: type-check
cd frontend && npm run type-check

# Linting completo del proyecto
ruff check ml/src/ backend/src/
cd frontend && npm run lint
```

---

## Decisiones de Diseño Clave

### ¿Por qué velas de 4h?
Equilibrio entre señal y ruido. Las velas de 1m tienen demasiado ruido, las diarias reducen el dataset de entrenamiento a pocas muestras.

### ¿Por qué clasificación binaria y no regresión de precio?
Predecir la dirección exacta del movimiento es más robusto y accionable que predecir el precio exacto. Un accuracy del 55–58% en clasificación binaria es estadísticamente significativo y potencialmente rentable.

### ¿Por qué Walk-Forward Validation?
Es el único método correcto para validar modelos en series temporales. KFold estándar mezcla datos del futuro con el pasado, produciendo métricas infladas (data leakage temporal).

### ¿Por qué ONNX para inferencia?
Desacopla el framework de entrenamiento (PyTorch/XGBoost) del framework de inferencia. El backend no necesita PyTorch instalado. La inferencia es ~3-5x más rápida.
