"""Tests for FastAPI Antigravity-Style Web Server & OpenAI /v1 Gateway."""

from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from nglai.config import NGLAIConfig
from nglai.engine.runtime import EngineRuntime
from nglai.server import create_app


def test_web_workspace_and_openai_v1_endpoints(tmp_path: Path) -> None:
    (tmp_path / "demo.py").write_text("def run():\n    return 42\n", encoding="utf-8")
    cfg = NGLAIConfig(base_dir=str(tmp_path))
    app = create_app(config=cfg, workspace_root=tmp_path)
    client = TestClient(app)

    # Index HTML
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "New-Gen-Local-AI" in r_root.text

    # Telemetry & Split Plan
    r_tel = client.get("/api/telemetry")
    assert r_tel.status_code == 200
    tel = r_tel.json()
    assert "split_plan" in tel
    assert "hardware" in tel

    # Models Catalog
    r_mod = client.get("/api/models")
    assert r_mod.status_code == 200
    assert len(r_mod.json()["models"]) >= 5

    # Workspace Explorer & AST Repo-Map
    r_tree = client.get("/api/workspace/tree")
    assert any(f["path"] == "demo.py" for f in r_tree.json()["files"])

    r_map = client.get("/api/workspace/repomap")
    assert "demo.py" in r_map.json()["repomap"]

    # OpenAI-Compatible /v1/models
    r_v1 = client.get("/v1/models")
    assert r_v1.status_code == 200
    assert r_v1.json()["object"] == "list"


def test_v1_proxy_auto_compacts_on_context_exceeded_and_returns_valid_json_errors(tmp_path: Path) -> None:
    cfg = NGLAIConfig(base_dir=str(tmp_path), context_size=8192)
    runtime = EngineRuntime(cfg)
    app = create_app(config=cfg, workspace_root=tmp_path, runtime_instance=runtime)
    client = TestClient(app)

    calls = {"count": 0}

    def fake_handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": 400,
                        "message": "request (8313 tokens) exceeds the available context size (8192 tokens), try increasing it",
                        "type": "exceed_context_size_error",
                    }
                },
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        )

    mock_transport = httpx.MockTransport(fake_handler)
    real_async_client = httpx.AsyncClient

    def patched_async_client(*args, **kwargs):
        kwargs["transport"] = mock_transport
        return real_async_client(*args, **kwargs)

    huge_messages = [{"role": "system", "content": "Você é um agente."}]
    for i in range(12):
        huge_messages.append({"role": "user", "content": f"Passo {i}: " + ("codigo " * 250)})
        huge_messages.append({"role": "assistant", "content": f"<think>raciocinio gigante {i} " + ("x" * 1000) + "</think>\nOk!"})

    with patch.object(runtime, "is_healthy", return_value=True), patch("nglai.server.httpx.AsyncClient", side_effect=patched_async_client):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3.6-35b-a3b-ud-iq3", "messages": huge_messages, "stream": True},
        )
        assert resp.status_code == 200
        assert calls["count"] == 2
        assert b'"content":"ok"' in resp.content

    # Now test a non-recoverable upstream 400 with newlines/quotes -> must return clean JSON, never broken SSE string!
    def bad_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b'Bad "payload"\nline 2 error')

    bad_transport = httpx.MockTransport(bad_handler)

    def patched_bad_client(*args, **kwargs):
        kwargs["transport"] = bad_transport
        return real_async_client(*args, **kwargs)

    with patch.object(runtime, "is_healthy", return_value=True), patch("nglai.server.httpx.AsyncClient", side_effect=patched_bad_client):
        resp_err = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3.6-35b-a3b-ud-iq3", "messages": [{"role": "user", "content": "oi"}], "stream": True},
        )
        assert resp_err.status_code == 400
        parsed = resp_err.json()
        assert "error" in parsed
        assert "line 2 error" in parsed["error"]["message"]
