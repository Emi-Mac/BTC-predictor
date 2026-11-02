#!/usr/bin/env bash
# Entorno de desarrollo de la rama de volatilidad: backend (8001), ML worker, base `btc_vol_dev`
# y frontend (3100). Requiere el contenedor btc_postgres (stack principal) en marcha.
#
#   infra/scripts/dev-volatility.sh up       # base, migraciones, modelos, datos, historial, backend y frontend
#   infra/scripts/dev-volatility.sh status
#   infra/scripts/dev-volatility.sh down     # detiene todo (conserva la base)
#   infra/scripts/dev-volatility.sh reset    # borra la base de desarrollo (pide confirmación)
#
# No toca la base del stack de la Fase 4 ni la de la sombra (btc-shadow).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE=(docker compose --project-directory "$ROOT" -f "$ROOT/infra/docker-compose.dev.yml")
PG=btc_postgres
DB="${DEV_POSTGRES_DB:-btc_vol_dev}"
API_PORT="${DEV_API_PORT:-8001}"
WEB_PORT="${DEV_WEB_PORT:-3100}"
DATASET="${SHADOW_DATASET:-artifacts/datasets/btcusdt-20200901-20261001-v1}"
AUX="${SHADOW_AUX:-artifacts/aux-sources/aux-20190901-20261001-v2}"
DEV_END="${SHADOW_DEV_END:-2026-10-01T00:00:00+00:00}"
REPLAY_CANDLES="${DEV_REPLAY_CANDLES:-330}"
LOGFILE="/tmp/btc-dev-frontend.log"

[[ "$DB" == *dev* ]] || { echo "La base de desarrollo debe llevar 'dev' en el nombre ($DB)"; exit 1; }
docker inspect "$PG" >/dev/null 2>&1 || { echo "Falta $PG: levanta primero el stack principal (postgres)."; exit 1; }
export DEV_POSTGRES_USER="${DEV_POSTGRES_USER:-$(docker exec "$PG" printenv POSTGRES_USER)}"
export DEV_POSTGRES_PASSWORD="${DEV_POSTGRES_PASSWORD:-$(docker exec "$PG" printenv POSTGRES_PASSWORD)}"
export DEV_POSTGRES_DB="$DB"

psql_dev() { docker exec "$PG" psql -U "$DEV_POSTGRES_USER" -d "$DB" -tA -c "$1"; }
tools() { "${COMPOSE[@]}" --profile tools run --rm -T ml-tools "$@"; }

ensure_database() {
  if [[ "$(docker exec "$PG" psql -U "$DEV_POSTGRES_USER" -d postgres -tA -c "select 1 from pg_database where datname='$DB'")" != "1" ]]; then
    echo "▶ Creando la base ${DB}…"
    docker exec "$PG" psql -U "$DEV_POSTGRES_USER" -d postgres -c "create database $DB" >/dev/null
  fi
}

frontend_pid() { lsof -ti "tcp:$WEB_PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true; }

start_frontend() {
  if [[ -n "$(frontend_pid)" ]]; then echo "▶ Frontend ya en marcha en el puerto $WEB_PORT."; return; fi
  echo "▶ Levantando el frontend en http://localhost:${WEB_PORT} (log: ${LOGFILE})…"
  # exec + redirecciones: el proceso queda desligado de la terminal y no retiene su salida.
  ( cd "$ROOT/frontend" && exec env NEXT_PUBLIC_API_URL="http://localhost:$API_PORT" \
      NEXT_PUBLIC_WS_URL="ws://localhost:$API_PORT" nohup npx next dev -p "$WEB_PORT" \
      >"$LOGFILE" 2>&1 </dev/null ) &
  disown
}

cmd_up() {
  ensure_database
  echo "▶ Construyendo imágenes y aplicando migraciones (001–009)…"
  "${COMPOSE[@]}" build migrate ml-worker
  "${COMPOSE[@]}" run --rm migrate
  if [[ "$(psql_dev "select count(*) from volatility_models where is_active")" == "0" ]]; then
    echo "▶ Entrenando, registrando y activando modelos (1d y 7d)…"
    for h in 6 42; do
      tools python -m src.volatility.train --horizon "$h" --register --dataset "$DATASET" --aux "$AUX" --development-end "$DEV_END" >/dev/null
    done
    for v in $(psql_dev "select version from volatility_models order by version"); do
      tools python -m src.volatility.activation --version "$v" --reason "entorno de desarrollo" >/dev/null
    done
  fi
  echo "▶ Levantando el ML worker (backfill inicial de ${DEV_HISTORY_DAYS:-200} días; unos minutos)…"
  "${COMPOSE[@]}" up -d ml-worker
  until [[ "$(psql_dev "select count(*) from volatility_forecasts" 2>/dev/null || echo 0)" -ge 2 ]]; do sleep 10; printf '.'; done; echo
  if [[ "$(psql_dev "select count(*) from volatility_forecasts")" -lt 100 ]]; then
    echo "▶ Reproduciendo $REPLAY_CANDLES velas de pronósticos para poblar historial y calibración…"
    tools python -m src.volatility.replay --candles "$REPLAY_CANDLES"
  fi
  echo "▶ Levantando el backend en http://localhost:$API_PORT …"
  "${COMPOSE[@]}" up -d backend
  start_frontend
  echo; echo "Listo → http://localhost:$WEB_PORT  (API: http://localhost:$API_PORT/docs)"
}

cmd_status() {
  "${COMPOSE[@]}" ps
  echo "── Frontend: $( [[ -n "$(frontend_pid)" ]] && echo "en marcha en el puerto $WEB_PORT" || echo detenido )"
  psql_dev "select horizon_candles, count(*), count(resolved_at) from volatility_forecasts group by 1 order by 1" || true
  curl -s -o /dev/null -w "API /health → %{http_code}\n" "http://localhost:$API_PORT/health" || true
}

cmd_down() {
  local pid; pid="$(frontend_pid)"; if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  "${COMPOSE[@]}" --profile tools down
}

cmd_reset() {
  read -r -p "Esto BORRA la base $DB (solo desarrollo). ¿Continuar? [escribe 'borrar'] " a
  [[ "$a" == "borrar" ]] || { echo Cancelado.; exit 1; }
  cmd_down
  docker exec "$PG" psql -U "$DEV_POSTGRES_USER" -d postgres -c "drop database if exists $DB" >/dev/null
  echo "Base $DB eliminada."
}

case "${1:-}" in
  up) cmd_up ;;
  status) cmd_status ;;
  down) cmd_down ;;
  reset) cmd_reset ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
