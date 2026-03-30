"""Tests for safe inputSchema access in MCP tool registration.

_convert_mcp_schema uses bare ``mcp_tool.inputSchema`` which raises
AttributeError if the MCP server returns a Tool object without that
attribute. Other call sites in the same file correctly use
``getattr(t, "inputSchema", None)``. This test verifies consistency.
"""

import types

import pytest


def _make_mcp_tool(name="test_tool", description="A test tool", input_schema=None, has_schema=True):
    """Build a minimal MCP Tool stub."""
    tool = types.SimpleNamespace(name=name, description=description)
    if has_schema:
        tool.inputSchema = input_schema
    return tool


class TestConvertMcpSchemaInputSchemaAccess:
    """tools/mcp_tool.py — _convert_mcp_schema()"""

    def test_tool_without_input_schema_attribute(self):
        """A Tool object with no inputSchema attribute should not crash."""
        from tools.mcp_tool import _convert_mcp_schema

        tool = _make_mcp_tool(has_schema=False)
        result = _convert_mcp_schema("my_server", tool)

        assert result["name"] == "mcp_my_server_test_tool"
        assert "parameters" in result

    def test_tool_with_none_input_schema(self):
        """A Tool with inputSchema=None should produce a valid schema."""
        from tools.mcp_tool import _convert_mcp_schema

        tool = _make_mcp_tool(input_schema=None)
        result = _convert_mcp_schema("my_server", tool)

        assert result["name"] == "mcp_my_server_test_tool"
        assert "parameters" in result

    def test_tool_with_valid_input_schema(self):
        """A Tool with a proper inputSchema should pass it through."""
        from tools.mcp_tool import _convert_mcp_schema

        schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        tool = _make_mcp_tool(input_schema=schema)
        result = _convert_mcp_schema("my_server", tool)

        assert result["parameters"]["properties"]["query"]["type"] == "string"

    def test_tool_name_sanitized(self):
        """Hyphens and dots in tool/server names should be replaced with underscores."""
        from tools.mcp_tool import _convert_mcp_schema

        tool = _make_mcp_tool(name="my-tool.v2")
        result = _convert_mcp_schema("my-server.io", tool)

        assert result["name"] == "mcp_my_server_io_my_tool_v2"


class TestSourceLineVerification:
    """Verify the actual source uses getattr for inputSchema."""

    @staticmethod
    def _read_source() -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(base, "tools", "mcp_tool.py")) as f:
            return f.read()

    def test_no_bare_input_schema_in_convert(self):
        """_convert_mcp_schema should not use bare mcp_tool.inputSchema."""
        src = self._read_source()
        # Find the function and check it uses getattr
        func_start = src.index("def _convert_mcp_schema")
        # Next function definition marks the end
        func_end = src.index("\ndef ", func_start + 1)
        func_body = src[func_start:func_end]

        assert "mcp_tool.inputSchema" not in func_body, (
            "_convert_mcp_schema still uses bare mcp_tool.inputSchema — "
            "use getattr(mcp_tool, 'inputSchema', None) instead"
        )
