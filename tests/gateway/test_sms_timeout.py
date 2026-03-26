"""Verify SmsAdapter sets a default timeout on its HTTP session."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from gateway.platforms.sms import SmsAdapter
    config = PlatformConfig(
        enabled=True,
        token="test-auth-token",
        extra={},
    )
    with patch.dict("os.environ", {
        "TWILIO_ACCOUNT_SID": "ACtest123",
        "TWILIO_AUTH_TOKEN": "test-auth-token",
        "TWILIO_PHONE_NUMBER": "+15551234567",
    }):
        return SmsAdapter(config)


class TestSmsSessionTimeout:
    @pytest.mark.asyncio
    async def test_connect_session_has_timeout(self):
        """connect() should create a ClientSession with a 30s total timeout."""
        import aiohttp
        adapter = _make_adapter()
        with patch("aiohttp.web.AppRunner") as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner_cls.return_value = mock_runner
            mock_site = AsyncMock()
            with patch("aiohttp.web.TCPSite", return_value=mock_site):
                result = await adapter.connect()

        assert result is True
        assert adapter._http_session is not None
        assert adapter._http_session.timeout.total == 30
        await adapter._http_session.close()

    @pytest.mark.asyncio
    async def test_send_uses_session_with_timeout(self):
        """send() should use the persistent session (which has a timeout)."""
        import aiohttp
        adapter = _make_adapter()

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"sid": "SM123"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        session.post = MagicMock(return_value=mock_resp)
        adapter._http_session = session

        result = await adapter.send("+15559876543", "Hello!")

        assert result.success is True
        assert session.timeout.total == 30
        await session.close()
