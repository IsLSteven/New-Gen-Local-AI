"""Persistent Project Memory (.nglai/memory.json) for long-term architectural recall."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


class ProjectMemory:
    """Stores architectural conventions, stack notes, and lessons learned per workspace."""

    def __init__(self, workspace_root: Path) -> None:
        self.root = workspace_root.resolve()
        self.mem_file = self.root / ".nglai" / "memory.json"

    def load_entries(self) -> List[Dict[str, str]]:
        if not self.mem_file.exists():
            return []
        try:
            data = json.loads(self.mem_file.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def add_note(self, note: str, category: str = "architecture") -> None:
        entries = self.load_entries()
        entries.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "category": category,
                "note": note.strip(),
            }
        )
        # Keep latest 50 high-signal notes
        entries = entries[-50:]
        self.mem_file.parent.mkdir(parents=True, exist_ok=True)
        self.mem_file.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")

    def clear(self) -> None:
        if self.mem_file.exists():
            self.mem_file.unlink()

    def render_for_prompt(self) -> str:
        entries = self.load_entries()
        if not entries:
            return ""
        lines = ["# Memória Persistente do Projeto"]
        for item in entries[-15:]:
            lines.append(f"- [{item.get('category', 'note')}] {item.get('note', '')}")
        return "\n".join(lines)
