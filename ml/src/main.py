"""Entrypoint del ml-worker (Dockerfile: `python -m src.main`)."""
from src.ingestion.run import main

if __name__ == "__main__":
    raise SystemExit(main())
