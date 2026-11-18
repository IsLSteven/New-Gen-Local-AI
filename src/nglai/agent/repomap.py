"""AST & Multi-Language Structural Repository Mapper (Opus/Argon-Style Context Engine)."""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

IGNORED_DIRS: Set[str] = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    "target",
    "out",
    ".next",
    ".nuxt",
    ".cache",
    ".nglai",
    "hot_ssd",
    "vault_hdd",
    "models",
    "engine_bin",
    "logs",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

CODE_EXTENSIONS: Set[str] = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".rs",
    ".go",
    ".cs",
    ".java",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".sql",
    ".toml",
    ".yaml",
    ".yml",
}


@dataclass
class FileSummary:
    rel_path: str
    language: str
    line_count: int
    symbols: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    score: float = 1.0


def _format_py_args(args: ast.arguments) -> str:
    names = [a.arg for a in args.args]
    if args.vararg:
        names.append(f"*{args.vararg.arg}")
    if args.kwarg:
        names.append(f"**{args.kwarg.arg}")
    if len(names) > 6:
        return ", ".join(names[:5]) + ", ..."
    return ", ".join(names)


def _extract_python_symbols(source: str) -> Tuple[List[str], List[str]]:
    symbols: List[str] = []
    imports: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _extract_generic_symbols(source, ".py"), []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.ClassDef):
            methods: List[str] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("__") or item.name == "__init__":
                        methods.append(f"{item.name}({_format_py_args(item.args)})")
            method_preview = f" [{', '.join(methods[:6])}]" if methods else ""
            symbols.append(f"class {node.name}{method_preview}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            symbols.append(f"{prefix} {node.name}({_format_py_args(node.args)})")
    return symbols, imports


_GENERIC_PATTERNS: List[Tuple[Set[str], re.Pattern]] = [
    (
        {".ts", ".tsx", ".js", ".jsx"},
        re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
            r"(?:(class|interface|type|enum|function)\s+([A-Za-z_]\w*)|const\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\()",
            re.MULTILINE,
        ),
    ),
    (
        {".rs"},
        re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(struct|enum|trait|fn|mod)\s+([A-Za-z_]\w*)", re.MULTILINE),
    ),
    (
        {".go"},
        re.compile(r"^\s*(?:(type)\s+([A-Za-z_]\w*)\s+(?:struct|interface)|(func)\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*))", re.MULTILINE),
    ),
    (
        {".cs", ".java", ".cpp", ".cc", ".c", ".h", ".hpp"},
        re.compile(r"^\s*(?:public|private|protected|internal|static|virtual|async|\s)*(class|interface|struct|enum)\s+([A-Za-z_]\w*)", re.MULTILINE),
    ),
]


def _extract_generic_symbols(source: str, ext: str) -> List[str]:
    found: List[str] = []
    for exts, pattern in _GENERIC_PATTERNS:
        if ext in exts:
            for match in pattern.finditer(source):
                groups = [g for g in match.groups() if g]
                if len(groups) >= 2:
                    found.append(f"{groups[0]} {groups[1]}")
                elif len(groups) == 1:
                    found.append(f"const {groups[0]}()")
                if len(found) >= 18:
                    break
    return found


def _tokenize_query(query: str) -> Set[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query.lower())
    stopwords = {
        "para", "como", "isso", "este", "essa", "fazer", "criar", "quero", "preciso",
        "the", "and", "for", "with", "that", "this", "from", "into", "uma", "dos", "das",
    }
    return {w for w in words if w not in stopwords}


class RepoMapper:
    """Builds a ranked, token-efficient structural map of the entire workspace."""

    def __init__(self, workspace_root: Path, max_files: int = 400) -> None:
        self.root = workspace_root.resolve()
        self.max_files = max_files

    def scan(self, query: str = "") -> List[FileSummary]:
        query_terms = _tokenize_query(query)
        summaries: List[FileSummary] = []
        scanned = 0

        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in sorted(dirnames) if d not in IGNORED_DIRS and not d.startswith(".git")]
            for fname in sorted(filenames):
                if scanned >= self.max_files:
                    break
                fpath = Path(dirpath) / fname
                ext = fpath.suffix.lower()
                if ext not in CODE_EXTENSIONS and fname not in {"Dockerfile", "Makefile", "README.md"}:
                    continue
                try:
                    if fpath.stat().st_size > 512_000:
                        continue
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                scanned += 1
                rel = fpath.relative_to(self.root).as_posix()
                lines = content.count("\n") + 1

                if ext == ".py":
                    symbols, imports = _extract_python_symbols(content)
                elif ext in CODE_EXTENSIONS:
                    symbols, imports = _extract_generic_symbols(content, ext), []
                else:
                    symbols, imports = [], []

                # Relevance scoring (filename hits + symbol hits + architectural hub bonus)
                score = 1.0
                rel_lower = rel.lower()
                sym_joined = " ".join(symbols).lower()
                for term in query_terms:
                    if term in rel_lower:
                        score += 5.0
                    if term in sym_joined:
                        score += 3.0
                if any(hub in rel_lower for hub in ("main", "server", "cli", "orchestrator", "config", "index", "app")):
                    score += 1.2

                summaries.append(
                    FileSummary(
                        rel_path=rel,
                        language=ext.lstrip(".") or "text",
                        line_count=lines,
                        symbols=symbols,
                        imports=imports,
                        score=score,
                    )
                )

        # Boost files imported by many other files (simplified PageRank centrality)
        stem_map: Dict[str, FileSummary] = {Path(s.rel_path).stem.lower(): s for s in summaries}
        for s in summaries:
            for imp in s.imports:
                tail = imp.split(".")[-1].lower()
                if tail in stem_map and stem_map[tail] is not s:
                    stem_map[tail].score += 0.8

        summaries.sort(key=lambda item: (-item.score, item.rel_path))
        return summaries

    def build_map_text(self, query: str = "", max_tokens: int = 2200) -> str:
        """Render the ranked AST Repo-Map within a strict token budget (~4 chars/token)."""
        summaries = self.scan(query=query)
        if not summaries:
            return "(Workspace vazio — nenhum arquivo de código detectado ainda.)"

        char_budget = max(600, max_tokens * 4)
        lines_out: List[str] = [f"# Mapa Estrutural do Repositório ({len(summaries)} arquivos indexados)"]
        used_chars = len(lines_out[0])

        for item in summaries:
            entry = f"- {item.rel_path} ({item.line_count}L)"
            if item.symbols:
                sym_str = "; ".join(item.symbols[:8])
                if len(item.symbols) > 8:
                    sym_str += f" ... (+{len(item.symbols) - 8})"
                entry += f": {sym_str}"

            if used_chars + len(entry) + 2 > char_budget:
                remaining = len(summaries) - len(lines_out) + 1
                if remaining > 0:
                    lines_out.append(f"... (+{remaining} arquivos adicionais omitidos pelo orçamento de contexto)")
                break

            lines_out.append(entry)
            used_chars += len(entry) + 1

        return "\n".join(lines_out)
