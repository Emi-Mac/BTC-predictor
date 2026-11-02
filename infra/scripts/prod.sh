#!/usr/bin/env bash
# Operación en vivo (datos reales, base `btcpredictor`): backend (8000), ML worker y frontend
# compilado (3000) de esta rama, con volatilidad habilitada.
#
#   infra/scripts/prod.sh up       # respaldo, migraciones, modelos, backend, worker y frontend
#   infra/scripts/prod.sh status
#   infra/scripts/prod.sh backup   # pg_dump de la base a ~/btc-backups
#   infra/scripts/prod.sh frontend # recompila y reinicia solo el frontend
#   infra/scripts/prod.sh down     # detiene backend, worker y frontend (conserva la base)
#
# Banderas y secretos: .env (ENVIRONMENT=production, VOLATILITY_*). No reproduce historial:
# los pronósticos y predicciones nacen del reloj real (nada retrospectivo).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE=(docker compose --project-directory "$ROOT" -p btc-predictor
  -f "$ROOT/infra/docker-compose.yml" -f "$ROOT/infra/docker-compose.prod.yml")
PG=btc_postgres
PGUSER="$(docker exec "$PG" printenv POSTGRES_USER)"
DB="$(docker exec "$PG" printenv POSTGRES_DB)"
WEB_PORT=3000
LOGFILE="/tmp/btc-prod-frontend.log"
DATASET="${SHADOW_DATASET:-artifacts/datasets/btcusdt-20200901-20261001-v1}"
AUX="${SHADOW_AUX:-artifacts/aux-sources/aux-20190901-20261001-v2}"
DEV_END="${SHADOW_DEV_END:-2026-10-01T00:00:00+00:00}"

psql_prod() { docker exec "$PG" psql -U "$PGUSER" -d "$DB" -tA -c "$1"; }
tools() { "${COMPOSE[@]}" --profile research run --rm -T ml-research "$@"; }
frontend_pid() { lsof -ti "tcp:$WEB_PORT" -sTCP:LISTEN 2>/dev/null | head -1 || true; }

cmd_backup() {
  mkdir -p "$HOME/btc-backups"
  local f="$HOME/btc-backups/${DB}-$(date -u +%Y%m%dT%H%M%SZ).dump"
  docker exec "$PG" pg_dump -U "$PGUSER" -Fc "$DB" >"$f"
  echo "▶ Respaldo: $f ($(du -h "$f" | cut -f1))"
}

cmd_frontend() {
  local pid; pid="$(frontend_pid)"
  if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; sleep 2; fi
  echo "▶ Compilando el frontend…"
  ( cd "$ROOT/frontend" && rm -rf .next \
      && NEXT_PUBLIC_API_URL="http://localhost:8000" NEXT_PUBLIC_WS_URL="ws://localhost:8000" npm run build >"$LOGFILE" 2>&1 ) \
    || { tail -20 "$LOGFILE"; exit 1; }
  echo "▶ Frontend en http://localhost:${WEB_PORT} (log: ${LOGFILE})"
  # exec + redirecciones: el proceso queda desligado de la terminal y no retiene su salida.
  ( cd "$ROOT/frontend" && exec nohup npx next start -p "$WEB_PORT" >>"$LOGFILE" 2>&1 </dev/null ) &
  disown
}

cmd_up() {
  cmd_backup
  echo "▶ Construyendo imágenes y aplicando migraciones (→ 009)…"
  "${COMPOSE[@]}" build backend ml-worker
  "${COMPOSE[@]}" --profile tools run --rm migrate
  if [[ "$(psql_prod "select count(*) from volatility_models where is_active")" == "0" ]]; then
    echo "▶ Entrenando, registrando y activando modelos de volatilidad (1d y 7d)…"
    for h in 6 42; do
      tools python -m src.volatility.train --horizon "$h" --register --dataset "$DATASET" --aux "$AUX" --development-end "$DEV_END" >/dev/null
    done
    for v in $(psql_prod "select version from volatility_models order by version"); do
      tools python -m src.volatility.activation --version "$v" --reason "puesta en producción" >/dev/null
    done
  fi
  echo "▶ Levantando ML worker y backend (modo producción)…"
  "${COMPOSE[@]}" up -d --no-deps --force-recreate ml-worker backend
  cmd_frontend
  echo; echo "Listo → http://localhost:$WEB_PORT  (API: http://localhost:8000/docs)"
}

cmd_status() {
  "${COMPOSE[@]}" ps
  echo "── Frontend: $( [[ -n "$(frontend_pid)" ]] && echo "en marcha en el puerto $WEB_PORT" || echo detenido )"
  psql_prod "select 'vol '||horizon_candles, count(*), count(resolved_at) from volatility_forecasts group by horizon_candles order by 1" || true
  psql_prod "select 'dir predictions', count(*) from predictions_log" || true
  curl -s -o /dev/null -w "API /health → %{http_code}\n" "http://localhost:8000/health" || true
}

cmd_down() {
  local pid; pid="$(frontend_pid)"; if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  "${COMPOSE[@]}" stop backend ml-worker
}

case "${1:-}" in
  up) cmd_up ;;
  status) cmd_status ;;
  backup) cmd_backup ;;
  frontend) cmd_frontend ;;
  down) cmd_down ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
