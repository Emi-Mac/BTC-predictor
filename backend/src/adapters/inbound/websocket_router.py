"""WebSocket endpoint: push de precio y predicciones en tiempo real (`/ws/live`).

Heartbeat cada 30s para mantener la conexión viva y detectar clientes caídos.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

log = structlog.get_logger(__name__)

router = APIRouter(tags=["WebSocket"])

HEARTBEAT_SECONDS = 30


class ConnectionManager:
    """Registro de conexiones WebSocket activas para difundir actualizaciones."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Envía `message` a todas las conexiones activas; descarta las que fallen."""
        payload = json.dumps(message)
        stale: list[WebSocket] = []
        for connection in list(self._connections):
            try:
                await connection.send_text(payload)
            except Exception:  # noqa: BLE001 - una conexión caída no debe romper el broadcast
                stale.append(connection)
        for connection in stale:
            self.disconnect(connection)


manager = ConnectionManager()


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
        with contextlib.suppress(Exception):
            await websocket.close()
