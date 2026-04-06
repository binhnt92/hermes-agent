"""Tests for platform lock release on connect() failure.

When a platform adapter acquires a scoped lock during connect() and then
fails (bad token, network error), the lock must be released so the next
gateway start isn't blocked with "already in use" errors.

Discord got this fix in PR #5302. Slack and Signal were missed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------

class TestSlackLockReleaseOnConnectFailure:
    """gateway/platforms/slack.py — connect() must release lock on exception."""

    @staticmethod
    def _read_source() -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(base, "gateway", "platforms", "slack.py")) as f:
            return f.read()

    def test_except_block_releases_lock(self):
        """The except block in connect() should call release_scoped_lock."""
        src = self._read_source()
        # Find the except block near the end of connect()
        start = src.index("async def connect(")
        end = src.index("\n    async def disconnect", start)
        connect_body = src[start:end]

        assert "release_scoped_lock" in connect_body, (
            "Slack connect() except block does not release the scoped lock — "
            "next gateway start will be blocked"
        )


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

class TestSignalLockReleaseOnConnectFailure:
    """gateway/platforms/signal.py — connect() must release lock on health check failure."""

    @staticmethod
    def _read_source() -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(base, "gateway", "platforms", "signal.py")) as f:
            return f.read()

    def test_health_check_failure_releases_lock(self):
        """Both return-False paths after health check must release the phone lock."""
        src = self._read_source()
        start = src.index("# Health check")
        end = src.index("self._running = True", start)
        health_block = src[start:end]

        assert health_block.count("_release_phone_lock") >= 2, (
            "Signal health check failure paths do not release the phone lock — "
            "both the non-200 and exception paths need _release_phone_lock()"
        )

    def test_release_phone_lock_method_exists(self):
        """_release_phone_lock helper should exist for reuse."""
        src = self._read_source()
        assert "def _release_phone_lock(self)" in src

    def test_health_check_failure_closes_client(self):
        """Both return-False paths must also close the httpx client."""
        src = self._read_source()
        start = src.index("# Health check")
        end = src.index("self._running = True", start)
        health_block = src[start:end]

        assert health_block.count("await self.client.aclose()") >= 2, (
            "Signal health check failure paths do not close the httpx client"
        )
