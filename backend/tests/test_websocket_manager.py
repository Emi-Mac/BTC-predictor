"""Tests de `ConnectionManager` (WebSocket `/ws/live`) con websockets simulados."""
from __future__ import annotations

import pytest

from src.adapters.inbound.websocket_router import ConnectionManager


class FakeWebSocket:
    def __init__(self, fail_on_send: bool = False) -> None:
        self.accepted = False
        self.sent: list[str] = []
        self._fail_on_send = fail_on_send

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, data: str) -> None:
        if self._fail_on_send:
            msg = "connection closed"
            raise RuntimeError(msg)
        self.sent.append(data)


@pytest.mark.asyncio
async def test_connect_accepts_and_registers_the_websocket_for_broadcast():
    manager = ConnectionManager()
    ws = FakeWebSocket()

    await manager.connect(ws)
    await manager.broadcast({"type": "heartbeat"})

    assert ws.accepted is True
    assert ws.sent == ['{"type": "heartbeat"}']


@pytest.mark.asyncio
async def test_broadcast_sends_the_same_payload_to_every_connection():
    manager = ConnectionManager()
    ws1, ws2 = FakeWebSocket(), FakeWebSocket()
    await manager.connect(ws1)
    await manager.connect(ws2)

    await manager.broadcast({"type": "prediction", "data": {"direction": "UP"}})

    assert ws1.sent == ws2.sent
    assert ws1.sent[0] == '{"type": "prediction", "data": {"direction": "UP"}}'


@pytest.mark.asyncio
async def test_broadcast_discards_connections_that_fail_to_send():
    manager = ConnectionManager()
    healthy = FakeWebSocket()
    broken = FakeWebSocket(fail_on_send=True)
    await manager.connect(healthy)
    await manager.connect(broken)

    await manager.broadcast({"type": "heartbeat"})  # No debe lanzar pese al fallo de `broken`.
    await manager.broadcast({"type": "prediction"})

    # `broken` se descarta tras el primer fallo: sólo recibe el primer intento.
    assert len(broken.sent) == 0
    assert len(healthy.sent) == 2


@pytest.mark.asyncio
async def test_disconnect_prevents_further_broadcasts():
    manager = ConnectionManager()
    ws = FakeWebSocket()
    await manager.connect(ws)

    manager.disconnect(ws)
    await manager.broadcast({"type": "heartbeat"})

    assert ws.sent == []
