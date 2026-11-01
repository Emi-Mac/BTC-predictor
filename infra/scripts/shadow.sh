#!/usr/bin/env bash
# Operación de la sombra prospectiva de volatilidad (F6).
#
#   infra/scripts/shadow.sh up        # levanta el stack, entrena y activa modelos, congela el protocolo
#   infra/scripts/shadow.sh status    # estado de contenedores, modelos y últimos pronósticos
#   infra/scripts/shadow.sh report    # evalúa los criterios con los datos acumulados
#   infra/scripts/shadow.sh api       # levanta la API de inspección (puerto 18100)
#   infra/scripts/shadow.sh down      # detiene (conserva la base)
#   infra/scripts/shadow.sh destroy   # borra contenedores Y base de datos (pide confirmación)
#
# `up` inicia el reloj de 12 semanas: el manifiesto (ml/artifacts/shadow/manifest.json) congela
# el protocolo, los modelos activos y el commit, y NO se puede reescribir.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT/infra/.shadow.env"
COMPOSE=(docker compose --project-directory "$ROOT" -f "$ROOT/infra/docker-compose.shadow.yml" --env-file "$ENV_FILE")

DATASET="${SHADOW_DATASET:-artifacts/datasets/btcusdt-20200901-20261001-v1}"
AUX="${SHADOW_AUX:-artifacts/aux-sources/aux-20190901-20261001-v2}"
DEV_END="${SHADOW_DEV_END:-2026-10-01T00:00:00+00:00}"
DB_USER="${SHADOW_POSTGRES_USER:-btcshadow}"
DB_NAME="${SHADOW_POSTGRES_DB:-btcshadow}"

# Commit evaluado (con «-dirty» si hay cambios sin confirmar); queda en el manifiesto.
SHADOW_GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]]; then SHADOW_GIT_COMMIT="${SHADOW_GIT_COMMIT}-dirty"; fi
export SHADOW_GIT_COMMIT

ensure_env() {
  if [[ ! -f "$ENV_FILE" ]]; then
    umask 077
    printf 'SHADOW_POSTGRES_PASSWORD=%s\n' "$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32)" >"$ENV_FILE"
    echo "Contraseña generada en $ENV_FILE (no se versiona)."
  fi
}

psql_shadow() {
  "${COMPOSE[@]}" exec -T postgres psql -U "$DB_USER" -d "$DB_NAME" -tA -c "$1"
}

tools() { "${COMPOSE[@]}" --profile tools run --rm -T ml-tools "$@"; }

cmd_up() {
  ensure_env
  echo "▶ Construyendo imágenes inmutables (código copiado, no montado)…"
  "${COMPOSE[@]}" build migrate ml-worker
  echo "▶ Base de datos y migraciones (hasta la 009)…"
  "${COMPOSE[@]}" up -d postgres
  "${COMPOSE[@]}" run --rm migrate
  local active
  active="$(psql_shadow "select count(*) from volatility_models where is_active")"
  if [[ "$active" == "0" ]]; then
    echo "▶ Entrenando y registrando modelos (1d y 7d) con el dataset congelado…"
    for horizon in 6 42; do
      tools python -m src.volatility.train --horizon "$horizon" --register \
        --dataset "$DATASET" --aux "$AUX" --development-end "$DEV_END" >/dev/null
    done
    echo "▶ Activando primario y respaldo de cada horizonte…"
    for version in $(psql_shadow "select version from volatility_models order by version"); do
      tools python -m src.volatility.activation --version "$version" --reason "sombra prospectiva F6" >/dev/null
    done
  else
    echo "▶ Ya hay modelos activos ($active); no se reentrena (los modelos deben permanecer congelados)."
  fi
  echo "▶ Congelando el protocolo de la sombra…"
  tools python -m src.volatility.shadow init || echo "(el protocolo ya estaba congelado)"
  echo "▶ Levantando el ML worker…"
  "${COMPOSE[@]}" up -d ml-worker
  echo "Listo. El worker hace el backfill inicial (varios minutos); revisa: $0 status"
}

cmd_status() {
  ensure_env
  "${COMPOSE[@]}" ps
  echo "── Modelos activos"
  psql_shadow "select version, kind, role, horizon_candles from volatility_models where is_active order by 4, 3"
  echo "── Pronósticos"
  psql_shadow "select horizon_candles, count(*), count(resolved_at), max(created_at) from volatility_forecasts group by 1 order by 1"
  echo "── Salud de ingesta"
  "${COMPOSE[@]}" exec -T ml-worker python -m src.ingestion.run check || true
}

cmd_report() { ensure_env; tools python -m src.volatility.shadow report; }

cmd_api() { ensure_env; "${COMPOSE[@]}" build migrate; "${COMPOSE[@]}" --profile api up -d backend; echo "API: http://localhost:${SHADOW_API_PORT:-18100}/docs"; }

cmd_down() { ensure_env; "${COMPOSE[@]}" --profile api --profile tools down; }

cmd_destroy() {
  ensure_env
  read -r -p "Esto BORRA la base de la sombra (el manifiesto en ml/artifacts/shadow se conserva). ¿Continuar? [escribe 'borrar'] " answer
  [[ "$answer" == "borrar" ]] || { echo "Cancelado."; exit 1; }
  "${COMPOSE[@]}" --profile api --profile tools down -v
}

case "${1:-}" in
  up) cmd_up ;;
  status) cmd_status ;;
  report) cmd_report ;;
  api) cmd_api ;;
  down) cmd_down ;;
  destroy) cmd_destroy ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
