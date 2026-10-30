#!/usr/bin/env bash
# ============================================================
# Script de setup inicial del proyecto BTC Predictor
# Ejecutar desde la raíz del proyecto: bash infra/scripts/setup.sh
# ============================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   BTC Predictor — Setup Inicial          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Verificar prerrequisitos ─────────────────────────────────
info "Verificando prerrequisitos..."

command -v docker    >/dev/null 2>&1 || error "Docker no encontrado. Instalar desde https://docs.docker.com/get-docker/"
command -v python3   >/dev/null 2>&1 || error "Python 3 no encontrado."
command -v node      >/dev/null 2>&1 || error "Node.js no encontrado."
command -v npm       >/dev/null 2>&1 || error "npm no encontrado."

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
NODE_VERSION=$(node --version | sed 's/v//' | cut -d'.' -f1)

[[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) -eq 1 ]] || error "Python 3.11+ requerido. Versión actual: $PYTHON_VERSION"
[[ $NODE_VERSION -ge 20 ]] || error "Node.js 20+ requerido. Versión actual: $NODE_VERSION"

success "Prerrequisitos OK (Python $PYTHON_VERSION, Node v$NODE_VERSION)"

# ── .env ────────────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    warning ".env creado desde .env.example. ¡EDITA las contraseñas antes de continuar!"
else
    info ".env ya existe, omitiendo."
fi

# ── Backend Python ───────────────────────────────────────────
info "Configurando entorno Python del backend..."
cd backend

if ! command -v uv &>/dev/null; then
    info "Instalando uv (gestor de paquetes rápido)..."
    pip install uv --quiet
fi

uv pip install --system -e ".[dev]" --quiet
success "Backend Python configurado."
cd ..

# ── ML Python ────────────────────────────────────────────────
info "Configurando entorno Python del ML worker..."
cd ml
uv pip install --system -e ".[dev]" --quiet
success "ML Python configurado."
cd ..

# ── Frontend Node ────────────────────────────────────────────
info "Instalando dependencias del frontend..."
cd frontend
npm install --silent
success "Frontend configurado."
cd ..

# ── Pre-commit ───────────────────────────────────────────────
if command -v pre-commit &>/dev/null; then
    info "Instalando pre-commit hooks..."
    pre-commit install
    success "Pre-commit hooks instalados."
else
    warning "pre-commit no encontrado. Instalar con: pip install pre-commit"
fi

# ── Docker: levantar PostgreSQL ──────────────────────────────
info "Levantando PostgreSQL con Docker..."
docker compose --project-directory . -f infra/docker-compose.yml up -d postgres

info "Esperando a que PostgreSQL esté listo..."
RETRIES=15
until docker compose --project-directory . -f infra/docker-compose.yml exec -T postgres pg_isready -U btcuser -d btcpredictor > /dev/null 2>&1; do
    RETRIES=$((RETRIES - 1))
    if [[ $RETRIES -le 0 ]]; then
        error "PostgreSQL no respondió después de 30 segundos."
    fi
    sleep 2
done
success "PostgreSQL listo."

# ── Alembic: migración inicial ───────────────────────────────
info "Aplicando migraciones de base de datos..."
cd backend
DATABASE_URL="postgresql+asyncpg://btcuser:changeme_in_production@localhost:5432/btcpredictor" \
    alembic upgrade head
success "Migraciones aplicadas (4 tablas creadas)."
cd ..

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   ✅ Setup completado exitosamente       ║"
echo "╠══════════════════════════════════════════╣"
echo "║  PostgreSQL:  localhost:5432             ║"
echo "║  Backend:     cd backend && uvicorn src.main:app --reload ║"
echo "║  Frontend:    cd frontend && npm run dev ║"
echo "║  PgAdmin:     docker compose --profile dev up ║"
echo "╚══════════════════════════════════════════╝"
echo ""
warning "Recuerda editar .env con contraseñas seguras antes de producción."

