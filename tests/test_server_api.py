"""Tests for FastAPI Antigravity-Style Web Server & OpenAI /v1 Gateway."""

from pathlib import Path

from fastapi.testclient import TestClient

from nglai.config import NGLAIConfig
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
