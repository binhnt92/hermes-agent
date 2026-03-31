"""Tests for atomic write in save_config_value.

save_config_value() previously used a bare open(path, 'w') + yaml.dump()
to write config.yaml. The 'w' mode truncates the file to zero bytes
immediately on open, before any data is written. If the process is
interrupted between truncation and completion of yaml.dump(), config.yaml
is left empty or partially written — losing all user configuration.

The fix replaces the bare write with atomic_yaml_write() which uses
temp file + fsync + os.replace, matching the gateway config write path.
"""

import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSaveConfigValueAtomic:
    """cli.py — save_config_value() must use atomic write."""

    def test_existing_config_preserved_on_new_key(self, tmp_path):
        """atomic_yaml_write preserves existing keys when adding new ones."""
        config_path = tmp_path / "config.yaml"
        original = {"api_key": "sk-secret", "model": "gpt-4"}
        config_path.write_text(yaml.dump(original))

        # Simulate the save_config_value flow: load, modify, atomic write
        config = yaml.safe_load(config_path.read_text()) or {}
        config["display"] = {"skin": "dark"}

        from utils import atomic_yaml_write
        atomic_yaml_write(config_path, config)

        result = yaml.safe_load(config_path.read_text())
        assert result["api_key"] == "sk-secret"
        assert result["model"] == "gpt-4"
        assert result["display"]["skin"] == "dark"

    def test_atomic_write_survives_if_file_exists(self, tmp_path):
        """atomic_yaml_write should not lose data even if called on existing file."""
        config_path = tmp_path / "config.yaml"
        original_data = {"provider": "openrouter", "model": "claude-3"}
        config_path.write_text(yaml.dump(original_data))

        from utils import atomic_yaml_write
        new_data = {**original_data, "display": {"show_reasoning": True}}
        atomic_yaml_write(config_path, new_data)

        result = yaml.safe_load(config_path.read_text())
        assert result["provider"] == "openrouter"
        assert result["display"]["show_reasoning"] is True

    def test_atomic_write_creates_new_file(self, tmp_path):
        """atomic_yaml_write should create the file if it doesn't exist."""
        config_path = tmp_path / "subdir" / "config.yaml"

        from utils import atomic_yaml_write
        atomic_yaml_write(config_path, {"new": "config"})

        assert config_path.exists()
        result = yaml.safe_load(config_path.read_text())
        assert result["new"] == "config"


class TestSourceLineVerification:
    """Verify cli.py save_config_value uses atomic_yaml_write."""

    @staticmethod
    def _read_source() -> str:
        import os
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "cli.py")) as f:
            return f.read()

    def test_no_bare_open_w_in_save_config_value(self):
        """save_config_value should not use open(path, 'w') for writing."""
        src = self._read_source()
        func_start = src.index("def save_config_value(")
        func_end = src.index("\ndef ", func_start + 1)
        func_body = src[func_start:func_end]

        assert "open(config_path, 'w')" not in func_body, (
            "cli.py save_config_value still uses bare open(path, 'w') — "
            "use atomic_yaml_write() instead"
        )

    def test_atomic_yaml_write_in_save_config_value(self):
        """save_config_value should use atomic_yaml_write."""
        src = self._read_source()
        func_start = src.index("def save_config_value(")
        func_end = src.index("\ndef ", func_start + 1)
        func_body = src[func_start:func_end]

        assert "atomic_yaml_write" in func_body, (
            "cli.py save_config_value does not use atomic_yaml_write"
        )
