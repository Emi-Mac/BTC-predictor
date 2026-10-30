-- ============================================================
-- BTC Predictor — Inicialización de PostgreSQL
-- Este script corre una sola vez al crear el contenedor.
-- ============================================================

-- Extensiones útiles
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- búsqueda de texto, útil para logs

-- Configuración de timezone
SET timezone = 'UTC';

