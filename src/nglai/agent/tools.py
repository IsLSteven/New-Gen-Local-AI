"""Autonomous Software Engineering Tools with Surgical Diffs, AST Validation & Safety Guards."""

from __future__ import annotations

import ast
import difflib
import fnmatch
import html
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from .repomap import IGNORED_DIRS

DANGEROUS_COMMAND_PATTERNS: List[re.Pattern] = [
    re.compile(r"\brm\s+-rf\s+(/|\*|~)", re.IGNORECASE),
    re.compile(r"\bdel\s+/[sqf]", re.IGNORECASE),
    re.compile(r"\bRemove-Item\b.*-Recurse.*-Force", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\s+-fdx?\b", re.IGNORECASE),
    re.compile(r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b|\breboot\b", re.IGNORECASE),
]


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: str
    diff: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def is_command_dangerous(command: str) -> bool:
    return any(p.search(command) for p in DANGEROUS_COMMAND_PATTERNS)


def validate_code_syntax(path: Path, content: str) -> Optional[str]:
    """Validate syntax for supported languages (.py, .json) immediately on write/edit."""
    ext = path.suffix.lower()
    if ext == ".py":
        try:
            ast.parse(content, filename=str(path))
            return None
        except SyntaxError as exc:
            return f"SyntaxError Python na linha {exc.lineno}, col {exc.offset}: {exc.msg}"
    if ext == ".json":
        try:
            json.loads(content)
            return None
        except json.JSONDecodeError as exc:
            return f"JSONDecodeError na linha {exc.lineno}, col {exc.colno}: {exc.msg}"
    return None


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Lê o conteúdo de um arquivo do projeto com numeração de linhas. Sempre leia antes de editar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho relativo ou absoluto do arquivo."},
                    "start_line": {"type": "integer", "description": "Linha inicial (1-indexada, opcional)."},
                    "end_line": {"type": "integer", "description": "Linha final inclusiva (opcional)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Cria um arquivo novo ou sobrescreve completamente um arquivo existente (com validação sintática automática).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho relativo ou absoluto do arquivo."},
                    "content": {"type": "string", "description": "Conteúdo completo a ser gravado."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edição cirúrgica de arquivo: substitui exatamente o bloco 'target' por 'replacement', gerando diff e validando sintaxe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho do arquivo a modificar."},
                    "target": {"type": "string", "description": "Trecho exato atual do arquivo (incluindo indentação)."},
                    "replacement": {"type": "string", "description": "Novo trecho substituto."},
                },
                "required": ["path", "target", "replacement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Busca expressões regulares ou texto exato em todos os arquivos do projeto.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Texto ou expressão regular a buscar."},
                    "file_glob": {"type": "string", "description": "Filtro opcional de arquivo, ex: '*.py' ou '*.ts'."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "Lista arquivos do workspace que correspondem a um padrão glob (ex: 'src/**/*.py').",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Padrão glob, padrão '*'."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Executa um comando no terminal do workspace (testes, linter, git, compiladores, scripts) e retorna stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Linha de comando a executar."},
                    "timeout_sec": {"type": "integer", "description": "Tempo limite em segundos (padrão 45)."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Pesquisa documentações ou erros recentes na web quando houver conexão disponível.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termos da pesquisa."},
                },
                "required": ["query"],
            },
        },
    },
]


class ToolExecutor:
    """Executes tool calls inside a sandboxed workspace boundary with human-in-the-loop safety."""

    def __init__(
        self,
        workspace_root: Path,
        command_approval_cb: Optional[Callable[[str, bool], bool]] = None,
        auto_approve_safe: bool = True,
        rollback_on_syntax_error: bool = False,
    ) -> None:
        self.root = workspace_root.resolve()
        self.command_approval_cb = command_approval_cb
        self.auto_approve_safe = auto_approve_safe
        self.rollback_on_syntax_error = rollback_on_syntax_error

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = Path(raw_path)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()
        # Prevent accidental traversal outside workspace unless explicitly inside workspace tree
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(
                f"Acesso bloqueado fora do workspace ({self.root}): {resolved}"
            ) from exc
        return resolved

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        try:
            dispatch = {
                "read_file": self._read_file,
                "write_file": self._write_file,
                "edit_file": self._edit_file,
                "grep_search": self._grep_search,
                "glob_files": self._glob_files,
                "run_command": self._run_command,
                "web_search": self._web_search,
            }
            if name not in dispatch:
                return ToolResult(name=name, ok=False, output=f"Ferramenta desconhecida: '{name}'.")
            return dispatch[name](arguments)
        except Exception as exc:
            return ToolResult(name=name, ok=False, output=f"Erro ao executar {name}: {type(exc).__name__}: {exc}")

    def _read_file(self, args: Dict[str, Any]) -> ToolResult:
        target = self._resolve_path(str(args["path"]))
        if not target.exists():
            return ToolResult("read_file", False, f"Arquivo não encontrado: {target.relative_to(self.root)}")
        if target.is_dir():
            return ToolResult("read_file", False, f"O caminho '{args['path']}' é um diretório, não um arquivo.")

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        start = max(1, int(args.get("start_line") or 1))
        end = min(total, int(args.get("end_line") or min(total, start + 399)))

        sliced = [f"{idx:4d}: {lines[idx - 1]}" for idx in range(start, end + 1)]
        header = f"Arquivo: {target.relative_to(self.root).as_posix()} (linhas {start}-{end} de {total})"
        return ToolResult("read_file", True, header + "\n" + "\n".join(sliced))

    def _write_file(self, args: Dict[str, Any]) -> ToolResult:
        target = self._resolve_path(str(args["path"]))
        content = str(args["content"])
        old_content = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""

        syntax_err = validate_code_syntax(target, content)
        if syntax_err and self.rollback_on_syntax_error:
            return ToolResult("write_file", False, f"Gravação abortada por erro de sintaxe -> {syntax_err}")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        rel = target.relative_to(self.root).as_posix()
        diff = "".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=3,
            )
        )
        msg = f"Arquivo gravado com sucesso: {rel} ({len(content.splitlines())} linhas)."
        if syntax_err:
            msg += f"\n⚠️ ALERTA SINTÁTICO IMEDIATO (corrija agora!): {syntax_err}"
        return ToolResult("write_file", syntax_err is None, msg, diff=diff)

    def _edit_file(self, args: Dict[str, Any]) -> ToolResult:
        target = self._resolve_path(str(args["path"]))
        if not target.exists():
            return ToolResult("edit_file", False, f"Arquivo não existe: {args['path']}")

        old_content = target.read_text(encoding="utf-8", errors="replace")
        needle = str(args["target"])
        replacement = str(args["replacement"])

        if needle not in old_content:
            # Helpful whitespace-normalized hint so compressed models recover immediately
            stripped_needle = needle.strip()
            hint = (
                "Dica: o trecho sem espaços nas bordas existe no arquivo; verifique a indentação exata com read_file."
                if stripped_needle and stripped_needle in old_content
                else "Use read_file para conferir as linhas exatas antes de chamar edit_file."
            )
            return ToolResult("edit_file", False, f"Bloco 'target' não encontrado em {args['path']}. {hint}")

        occurrences = old_content.count(needle)
        new_content = old_content.replace(needle, replacement, 1)

        syntax_err = validate_code_syntax(target, new_content)
        if syntax_err and self.rollback_on_syntax_error:
            return ToolResult("edit_file", False, f"Edição revertida por erro de sintaxe -> {syntax_err}")

        target.write_text(new_content, encoding="utf-8")
        rel = target.relative_to(self.root).as_posix()
        diff = "".join(
            difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=3,
            )
        )
        note = f" (1ª de {occurrences} ocorrências substituída)" if occurrences > 1 else ""
        msg = f"Edição cirúrgica aplicada em {rel}{note}."
        if syntax_err:
            msg += f"\n⚠️ ALERTA SINTÁTICO IMEDIATO (corrija agora!): {syntax_err}"
        return ToolResult("edit_file", syntax_err is None, msg, diff=diff)

    def _grep_search(self, args: Dict[str, Any]) -> ToolResult:
        pattern_str = str(args["pattern"])
        file_glob = str(args.get("file_glob") or "*")
        try:
            regex = re.compile(pattern_str, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern_str), re.IGNORECASE)

        hits: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in sorted(dirnames) if d not in IGNORED_DIRS]
            for fname in sorted(filenames):
                if not fnmatch.fnmatch(fname, file_glob):
                    continue
                fpath = Path(dirpath) / fname
                try:
                    if fpath.stat().st_size > 1_000_000:
                        continue
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                rel = fpath.relative_to(self.root).as_posix()
                for idx, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        hits.append(f"{rel}:{idx}: {line.strip()[:180]}")
                        if len(hits) >= 80:
                            return ToolResult(
                                "grep_search",
                                True,
                                "\n".join(hits) + "\n... (truncado em 80 resultados)",
                            )

        if not hits:
            return ToolResult("grep_search", True, f"Nenhuma ocorrência encontrada para '{pattern_str}'.")
        return ToolResult("grep_search", True, "\n".join(hits))

    def _glob_files(self, args: Dict[str, Any]) -> ToolResult:
        pattern = str(args.get("pattern") or "*")
        matched: List[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in sorted(dirnames) if d not in IGNORED_DIRS]
            for fname in sorted(filenames):
                rel = (Path(dirpath) / fname).relative_to(self.root).as_posix()
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fname, pattern):
                    matched.append(rel)
                    if len(matched) >= 250:
                        break
        if not matched:
            return ToolResult("glob_files", True, f"Nenhum arquivo encontrado para '{pattern}'.")
        return ToolResult("glob_files", True, "\n".join(matched))

    def _run_command(self, args: Dict[str, Any]) -> ToolResult:
        command = str(args["command"]).strip()
        timeout_sec = max(3, min(300, int(args.get("timeout_sec") or 45)))
        dangerous = is_command_dangerous(command)

        if dangerous or not self.auto_approve_safe:
            if self.command_approval_cb is None:
                return ToolResult(
                    "run_command",
                    False,
                    f"Comando bloqueado pela guarda de segurança (potencialmente destrutivo): '{command}'",
                )
            approved = self.command_approval_cb(command, dangerous)
            if not approved:
                return ToolResult("run_command", False, "Execução cancelada pelo usuário.")

        shell_cmd = (
            ["powershell", "-NoProfile", "-Command", command]
            if os.name == "nt"
            else ["bash", "-lc", command]
        )
        proc = subprocess.run(
            shell_cmd,
            cwd=str(self.root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        combined = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
        if len(combined) > 12_000:
            combined = combined[:6000] + "\n... [saída intermediária truncada] ...\n" + combined[-5000:]
        status_hdr = f"[exit_code={proc.returncode}]"
        return ToolResult("run_command", proc.returncode == 0, f"{status_hdr}\n{combined}".strip())

    def _web_search(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args["query"]).strip()
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (New-Gen-Local-AI Research Agent)"}
        try:
            resp = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
            resp.raise_for_status()
            snippets = re.findall(
                r'<a[^>]+class="result__a"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                resp.text,
                flags=re.DOTALL,
            )
            results: List[str] = []
            for title_raw, snip_raw in snippets[:6]:
                title = html.unescape(re.sub(r"<[^>]+>", "", title_raw)).strip()
                snip = html.unescape(re.sub(r"<[^>]+>", "", snip_raw)).strip()
                results.append(f"- {title}: {snip}")
            if not results:
                return ToolResult("web_search", True, f"Sem resultados extraídos para '{query}'.")
            return ToolResult("web_search", True, "\n".join(results))
        except Exception as exc:
            return ToolResult("web_search", False, f"Pesquisa web indisponível offline ({exc}).")
