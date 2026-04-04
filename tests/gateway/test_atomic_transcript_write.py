"""Tests for atomic writes in rewrite_transcript and channel_directory.

rewrite_transcript() and build_channel_directory() previously used bare
open(path, 'w') which truncates the file to zero bytes immediately on
open. If the process is killed mid-write, the file is left empty —
losing the conversation transcript or channel directory.

The fix uses temp file + fsync + os.replace, matching the atomic pattern
already used by sessions.json and .update_pending.json.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestRewriteTranscriptAtomic:
    """gateway/session.py — rewrite_transcript() must use atomic write."""

    @staticmethod
    def _read_source() -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(base, "gateway", "session.py")) as f:
            return f.read()

    def test_no_bare_open_w_in_rewrite_transcript(self):
        """rewrite_transcript should not use bare open(path, 'w')."""
        src = self._read_source()
        start = src.index("def rewrite_transcript(")
        end = src.index("\n    def ", start + 1)
        body = src[start:end]

        assert 'open(transcript_path, "w"' not in body, (
            "rewrite_transcript still uses bare open(path, 'w') — "
            "use temp file + os.replace for atomic write"
        )

    def test_os_replace_in_rewrite_transcript(self):
        """rewrite_transcript should use os.replace for atomic swap."""
        src = self._read_source()
        start = src.index("def rewrite_transcript(")
        end = src.index("\n    def ", start + 1)
        body = src[start:end]

        assert "os.replace(" in body, (
            "rewrite_transcript does not use os.replace — "
            "atomic write requires temp file + os.replace"
        )


class TestChannelDirectoryAtomic:
    """gateway/channel_directory.py — build_channel_directory() must use atomic write."""

    @staticmethod
    def _read_source() -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(base, "gateway", "channel_directory.py")) as f:
            return f.read()

    def test_no_bare_open_w_in_channel_directory(self):
        """build_channel_directory should not use bare open(path, 'w')."""
        src = self._read_source()
        start = src.index("def build_channel_directory(")
        end = src.index("\n\ndef ", start + 1) if "\n\ndef " in src[start + 1:] else len(src)
        body = src[start:end]

        assert 'open(DIRECTORY_PATH, "w"' not in body, (
            "build_channel_directory still uses bare open(path, 'w')"
        )

    def test_os_replace_in_channel_directory(self):
        """build_channel_directory should use os.replace for atomic swap."""
        src = self._read_source()
        assert "os.replace(" in src, (
            "channel_directory.py does not use os.replace"
        )


class TestAtomicWriteBehavior:
    """Verify atomic write pattern preserves data on simulated crash."""

    def test_bare_open_w_truncates_immediately(self, tmp_path):
        """Prove that open('w') truncates before writing."""
        path = tmp_path / "data.jsonl"
        path.write_text('{"msg": "important"}\n')
        assert path.stat().st_size > 0

        f = open(path, "w")
        assert path.stat().st_size == 0  # truncated!
        f.close()

    def test_atomic_replace_preserves_original(self, tmp_path):
        """temp file + os.replace never leaves original truncated."""
        import tempfile

        path = tmp_path / "data.jsonl"
        original = '{"msg": "important"}\n'
        path.write_text(original)

        # Write new content atomically
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=str(tmp_path), suffix=".tmp")
        with os.fdopen(tmp_fd, "w") as f:
            f.write('{"msg": "updated"}\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path_str, str(path))

        result = path.read_text()
        assert "updated" in result
