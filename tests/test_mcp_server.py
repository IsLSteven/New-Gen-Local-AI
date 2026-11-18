"""Tests for native Google Antigravity MCP Stdio Server."""

from pathlib import Path

from nglai.config import NGLAIConfig
from nglai.mcp_server import handle_mcp_request


def test_mcp_initialize_and_list_tools(tmp_path: Path) -> None:
    cfg = NGLAIConfig(base_dir=str(tmp_path))

    init_resp = handle_mcp_request(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        cfg=cfg,
    )
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "new-gen-local-ai"

    tools_resp = handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, cfg=cfg)
    assert tools_resp is not None
    names = [t["name"] for t in tools_resp["result"]["tools"]]
    assert "ask_local_ai" in names
    assert "run_local_coding_agent" in names
    assert "get_local_ai_status" in names

    status_call = handle_mcp_request(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_local_ai_status", "arguments": {}}},
        cfg=cfg,
    )
    assert status_call is not None
    assert status_call["result"]["isError"] is False
