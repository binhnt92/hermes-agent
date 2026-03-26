"""Tests for the startup allowlist warning check in gateway/run.py."""

import os
import logging
from unittest.mock import patch, MagicMock

import pytest


def _run_startup_check():
    """Execute only the startup allowlist warning logic from HermesGateway.run().

    Returns True if the warning was emitted, False otherwise.
    """
    _any_allowlist = any(
        os.getenv(v)
        for v in ("TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
                   "WHATSAPP_ALLOWED_USERS", "SLACK_ALLOWED_USERS",
                   "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS",
                   "EMAIL_ALLOWED_USERS",
                   "SMS_ALLOWED_USERS", "MATTERMOST_ALLOWED_USERS",
                   "MATRIX_ALLOWED_USERS", "DINGTALK_ALLOWED_USERS",
                   "GATEWAY_ALLOWED_USERS")
    )
    _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in ("true", "1", "yes") or any(
        os.getenv(v, "").lower() in ("true", "1", "yes")
        for v in ("TELEGRAM_ALLOW_ALL_USERS", "DISCORD_ALLOW_ALL_USERS",
                   "WHATSAPP_ALLOW_ALL_USERS", "SLACK_ALLOW_ALL_USERS",
                   "SIGNAL_ALLOW_ALL_USERS", "EMAIL_ALLOW_ALL_USERS",
                   "SMS_ALLOW_ALL_USERS", "MATTERMOST_ALLOW_ALL_USERS",
                   "MATRIX_ALLOW_ALL_USERS", "DINGTALK_ALLOW_ALL_USERS")
    )
    return not _any_allowlist and not _allow_all


class TestAllowlistStartupCheck:
    """Verify the startup warning is suppressed when allowlists are configured."""

    def test_no_config_emits_warning(self):
        """With no env vars set, the warning should fire."""
        with patch.dict(os.environ, {}, clear=True):
            assert _run_startup_check() is True

    def test_signal_group_allowed_users_suppresses_warning(self):
        """SIGNAL_GROUP_ALLOWED_USERS should be recognised as an allowlist."""
        with patch.dict(os.environ, {"SIGNAL_GROUP_ALLOWED_USERS": "user1"}, clear=True):
            assert _run_startup_check() is False

    def test_signal_allowed_users_suppresses_warning(self):
        """SIGNAL_ALLOWED_USERS should also suppress the warning."""
        with patch.dict(os.environ, {"SIGNAL_ALLOWED_USERS": "user1"}, clear=True):
            assert _run_startup_check() is False

    def test_telegram_allow_all_users_suppresses_warning(self):
        """A per-platform ALLOW_ALL var should suppress the warning."""
        with patch.dict(os.environ, {"TELEGRAM_ALLOW_ALL_USERS": "true"}, clear=True):
            assert _run_startup_check() is False

    def test_discord_allow_all_users_suppresses_warning(self):
        """DISCORD_ALLOW_ALL_USERS=1 should suppress the warning."""
        with patch.dict(os.environ, {"DISCORD_ALLOW_ALL_USERS": "1"}, clear=True):
            assert _run_startup_check() is False

    def test_gateway_allow_all_users_suppresses_warning(self):
        """GATEWAY_ALLOW_ALL_USERS should still work."""
        with patch.dict(os.environ, {"GATEWAY_ALLOW_ALL_USERS": "yes"}, clear=True):
            assert _run_startup_check() is False
