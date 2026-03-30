"""Tests for Signal httpx client cleanup on failed health check.

When Signal's connect() creates an httpx.AsyncClient and the health check
then fails (non-200 or network error), both return-False paths must close
the client to avoid leaking TCP connections and file descriptors.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_adapter():
    """Build a minimal SignalAdapter with mocked internals."""
    from gateway.platforms.signal import SignalAdapter

    adapter = SignalAdapter.__new__(SignalAdapter)
    adapter.http_url = "http://localhost:8080"
    adapter.account = "+1234567890"
    adapter._closing = False
    adapter._running = False
    adapter.client = None
    adapter._phone_lock_identity = None
    adapter._session_lock_identity = None
    adapter.platform = MagicMock()
    adapter.platform.value = "signal"
    return adapter


class TestSignalClientLeakOnHealthCheckFailure:
    """gateway/platforms/signal.py — connect() health check paths"""

    def test_client_closed_on_non_200_health_check(self):
        """A non-200 health check should close the httpx client."""
        adapter = _make_adapter()

        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        async def run():
            with patch("gateway.platforms.signal.httpx.AsyncClient",
                       return_value=mock_client):
                with patch("gateway.status.acquire_scoped_lock",
                           return_value=(True, None)):
                    result = await adapter.connect()

            assert result is False
            mock_client.aclose.assert_called_once()
            assert adapter.client is None

        asyncio.run(run())

    def test_client_closed_on_connection_error(self):
        """A connection error during health check should close the client."""
        adapter = _make_adapter()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
        mock_client.aclose = AsyncMock()

        async def run():
            with patch("gateway.platforms.signal.httpx.AsyncClient",
                       return_value=mock_client):
                with patch("gateway.status.acquire_scoped_lock",
                           return_value=(True, None)):
                    result = await adapter.connect()

            assert result is False
            mock_client.aclose.assert_called_once()
            assert adapter.client is None

        asyncio.run(run())

    def test_client_kept_on_successful_health_check(self):
        """A 200 health check should keep the client open."""
        adapter = _make_adapter()

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        async def run():
            with patch("gateway.platforms.signal.httpx.AsyncClient",
                       return_value=mock_client):
                with patch("gateway.status.acquire_scoped_lock",
                           return_value=(True, None)):
                    with patch("asyncio.create_task", return_value=MagicMock()):
                        result = await adapter.connect()

            assert result is True
            mock_client.aclose.assert_not_called()
            assert adapter.client is mock_client

        asyncio.run(run())


class TestSourceLineVerification:
    """Verify signal.py has aclose() on both early-return paths."""

    @staticmethod
    def _read_source() -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(base, "gateway", "platforms", "signal.py")) as f:
            return f.read()

    def test_aclose_present_in_connect(self):
        src = self._read_source()
        assert "await self.client.aclose()" in src, (
            "signal.py connect() missing await self.client.aclose() "
            "on health check failure paths"
        )
