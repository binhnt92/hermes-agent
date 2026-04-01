"""Tests for fallback state reset in reset_session_state.

When the primary model fails and Hermes falls back to an alternative
provider, _fallback_index and _fallback_activated are set. But
reset_session_state() — called between gateway conversations to reuse
the cached agent — never reset these fields. This means every subsequent
conversation stays on the fallback model permanently, and once the
fallback chain is exhausted, fallback protection is disabled forever.
"""

import pytest


class TestFallbackResetOnSessionReset:
    """run_agent.py — reset_session_state() must reset fallback fields."""

    @staticmethod
    def _make_agent():
        """Build a minimal AIAgent bypassing __init__."""
        from run_agent import AIAgent
        agent = AIAgent.__new__(AIAgent)
        # Set minimal attributes needed by reset_session_state
        agent.session_total_tokens = 100
        agent.session_input_tokens = 50
        agent.session_output_tokens = 50
        agent.session_prompt_tokens = 50
        agent.session_completion_tokens = 50
        agent.session_cache_read_tokens = 0
        agent.session_cache_write_tokens = 0
        agent.session_reasoning_tokens = 0
        agent.session_api_calls = 5
        agent.session_estimated_cost_usd = 0.01
        agent.session_cost_status = "estimated"
        agent.session_cost_source = "pricing"
        agent._user_turn_count = 3
        agent._fallback_index = 2
        agent._fallback_activated = True
        agent._fallback_chain = [
            {"model": "gpt-4o", "provider": "openai"},
            {"model": "claude-3", "provider": "anthropic"},
        ]
        return agent

    def test_fallback_index_reset_to_zero(self):
        """_fallback_index must be 0 after reset so fallback starts fresh."""
        agent = self._make_agent()
        assert agent._fallback_index == 2  # pre-condition

        agent.reset_session_state()

        assert agent._fallback_index == 0

    def test_fallback_activated_reset_to_false(self):
        """_fallback_activated must be False after reset."""
        agent = self._make_agent()
        assert agent._fallback_activated is True  # pre-condition

        agent.reset_session_state()

        assert agent._fallback_activated is False

    def test_fallback_chain_preserved(self):
        """The fallback chain itself should not be cleared — only the pointer."""
        agent = self._make_agent()

        agent.reset_session_state()

        assert len(agent._fallback_chain) == 2
        assert agent._fallback_chain[0]["model"] == "gpt-4o"

    def test_token_counters_also_reset(self):
        """Verify token counters are still reset (no regression)."""
        agent = self._make_agent()

        agent.reset_session_state()

        assert agent.session_total_tokens == 0
        assert agent.session_api_calls == 0
        assert agent.session_estimated_cost_usd == 0.0


class TestSourceLineVerification:
    """Verify the source has fallback reset in reset_session_state."""

    @staticmethod
    def _read_reset_method() -> str:
        import os
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "run_agent.py")) as f:
            src = f.read()
        start = src.index("def reset_session_state(self)")
        end = src.index("\n    def ", start + 1)
        return src[start:end]

    def test_fallback_index_reset_present(self):
        body = self._read_reset_method()
        assert "self._fallback_index = 0" in body, (
            "reset_session_state() missing self._fallback_index = 0"
        )

    def test_fallback_activated_reset_present(self):
        body = self._read_reset_method()
        assert "self._fallback_activated = False" in body, (
            "reset_session_state() missing self._fallback_activated = False"
        )
