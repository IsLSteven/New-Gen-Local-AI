"""Tests for AST Repo-Map Indexer, Surgical Tools, AST Syntax Guard & Destructive Command Guard."""

from pathlib import Path

from nglai.agent.repomap import RepoMapper
from nglai.agent.tools import ToolExecutor, is_command_dangerous


def test_ast_repomap_indexes_python_and_typescript(tmp_path: Path) -> None:
    (tmp_path / "auth_service.py").write_text(
        "class AuthManager:\n    def verify_token(self, token: str) -> bool:\n        return True\n",
        encoding="utf-8",
    )
    (tmp_path / "ui.ts").write_text(
        "export interface UserProfile { id: string }\nexport async function fetchUser(id: string) {}\n",
        encoding="utf-8",
    )

    mapper = RepoMapper(tmp_path)
    map_text = mapper.build_map_text(query="AuthManager verify_token")
    assert "auth_service.py" in map_text
    assert "class AuthManager" in map_text
    assert "verify_token" in map_text
    assert "ui.ts" in map_text
    # Ranked first due to query hit
    assert map_text.index("auth_service.py") < map_text.index("ui.ts")


def test_surgical_edit_and_syntax_validation(tmp_path: Path) -> None:
    executor = ToolExecutor(workspace_root=tmp_path)

    res_w = executor.execute(
        "write_file",
        {"path": "calc.py", "content": "def add(a, b):\n    return a - b\n"},
    )
    assert res_w.ok is True

    res_e = executor.execute(
        "edit_file",
        {
            "path": "calc.py",
            "target": "    return a - b",
            "replacement": "    return a + b",
        },
    )
    assert res_e.ok is True
    assert res_e.diff is not None
    assert "+    return a + b" in res_e.diff

    # Broken Python syntax triggers immediate warning & ok=False
    res_bad = executor.execute(
        "edit_file",
        {
            "path": "calc.py",
            "target": "    return a + b",
            "replacement": "    return a + (",
        },
    )
    assert res_bad.ok is False
    assert "SyntaxError" in res_bad.output


def test_dangerous_command_guard() -> None:
    assert is_command_dangerous("rm -rf /") is True
    assert is_command_dangerous("git reset --hard HEAD~1") is True
    assert is_command_dangerous("DROP TABLE users;") is True
    assert is_command_dangerous("pytest -q") is False
