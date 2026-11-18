"""Compound AI Orchestrator (Plan -> Deep Thinking -> Surgical Act -> Auto-Verify -> Critique)."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from ..config import NGLAIConfig
from .compactor import compact_conversation, estimate_history_tokens
from .memory import ProjectMemory
from .repomap import RepoMapper
from .tools import TOOL_SCHEMAS, ToolExecutor, ToolResult


@dataclass
class AgentEvent:
    """Structured real-time event emitted during an autonomous coding turn."""

    kind: str  # "thinking", "content", "tool_call", "tool_result", "compacted", "done", "error"
    text: str = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    diff: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


_TOOL_CALL_XML_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _repair_and_parse_json(raw: str) -> Dict[str, Any]:
    """Resilient JSON parser that heals minor trailing-comma formatting quirks in 3-bit quants."""
    cleaned = raw.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        healed = re.sub(r",\s*([}\]])", r"\1", cleaned)
        return json.loads(healed)


def extract_thinking_and_fallback_tools(content: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Separate <think>...</think> reasoning blocks and recover inline <tool_call>{...}</tool_call>
    invocations when compressed models output Qwen/Hermes XML tags instead of API tool_calls.
    """
    thoughts: List[str] = []
    for match in _THINK_BLOCK_RE.finditer(content):
        thoughts.append(match.group(1).strip())
    visible = _THINK_BLOCK_RE.sub("", content)

    recovered_calls: List[Dict[str, Any]] = []
    for idx, match in enumerate(_TOOL_CALL_XML_RE.finditer(visible)):
        payload_str = match.group(1)
        try:
            parsed = _repair_and_parse_json(payload_str)
            if "name" in parsed:
                args = parsed.get("arguments") or parsed.get("parameters") or {}
                if isinstance(args, str):
                    args = _repair_and_parse_json(args)
                recovered_calls.append(
                    {
                        "id": f"call_xml_{uuid.uuid4().hex[:8]}_{idx}",
                        "type": "function",
                        "function": {
                            "name": str(parsed["name"]),
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
        except Exception:
            continue

    visible = _TOOL_CALL_XML_RE.sub("", visible).strip()
    return "\n\n".join(thoughts), visible, recovered_calls


def build_compound_system_prompt(
    workspace_root: Path,
    repo_map_text: str,
    memory_text: str,
    enable_thinking: bool = True,
) -> str:
    thinking_directive = (
        "Antes de agir em tarefas complexas, estruture seu raciocínio arquitetural dentro de `<think>...</think>`: "
        "mapeie arquivos afetados, dependências, casos de borda e plano de verificação."
        if enable_thinking
        else "Seja objetivo, preciso e cirúrgico nas chamadas de ferramentas."
    )
    sections = [
        "Você é o **New-Gen-Local-AI**, um Engenheiro de Software Sênior Autônomo rodando 100% localmente "
        "sobre arquitetura Compound AI (Plan -> Deep Thinking -> Surgical Edit -> Auto-Verify -> Self-Critique).",
        f"Workspace ativo: `{workspace_root}`",
        "## Regras de Ouro de Engenharia Nível Fronteira\n"
        "1. **Nunca adivinhe código sem ler:** Use `read_file` ou `grep_search` antes de editar um arquivo existente.\n"
        "2. **Edições Cirúrgicas:** Prefira `edit_file` passando o trecho exato com indentação idêntica. Use `write_file` para arquivos novos.\n"
        "3. **Auto-Verificação Obrigatória:** Após modificar código, rode testes/linter/verificação via `run_command` (ex: `pytest`, `python -m py_compile`) "
        "e, se houver erro ou alerta sintático, corrija imediatamente antes de concluir.\n"
        "4. **Idioma:** Responda sempre no mesmo idioma do usuário (Português por padrão), com código limpo, modular e bem tipado.\n"
        f"5. **Raciocínio Estruturado:** {thinking_directive}",
        repo_map_text,
    ]
    if memory_text:
        sections.append(memory_text)
    return "\n\n".join(s for s in sections if s)


class CompoundCodingAgent:
    """Stateful autonomous software engineering agent backed by local OpenAI-compatible engine."""

    def __init__(
        self,
        config: NGLAIConfig,
        workspace_root: Optional[Path] = None,
        command_approval_cb: Optional[Callable[[str, bool], bool]] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.config = config
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.mapper = RepoMapper(self.workspace_root)
        self.memory = ProjectMemory(self.workspace_root)
        self.executor = ToolExecutor(
            workspace_root=self.workspace_root,
            command_approval_cb=command_approval_cb,
            auto_approve_safe=config.auto_approve_safe_commands,
        )
        self._client = http_client or httpx.Client(timeout=httpx.Timeout(30.0, read=600.0))
        self.history: List[Dict[str, Any]] = []

    @property
    def completions_url(self) -> str:
        return f"http://{self.config.engine_host}:{self.config.engine_port}/v1/chat/completions"

    def reset_session(self) -> None:
        self.history.clear()

    def _refresh_system_prompt(self, latest_user_query: str) -> None:
        repo_map = self.mapper.build_map_text(
            query=latest_user_query,
            max_tokens=self.config.repomap_max_tokens,
        )
        mem_text = self.memory.render_for_prompt()
        sys_prompt = build_compound_system_prompt(
            workspace_root=self.workspace_root,
            repo_map_text=repo_map,
            memory_text=mem_text,
            enable_thinking=self.config.enable_thinking,
        )
        if self.history and self.history[0].get("role") == "system":
            self.history[0]["content"] = sys_prompt
        else:
            self.history.insert(0, {"role": "system", "content": sys_prompt})

    def run_turn(
        self,
        user_prompt: str,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
    ) -> str:
        """Execute a complete autonomous multi-step ReAct + Verification loop."""
        emit = on_event or (lambda _ev: None)
        self._refresh_system_prompt(user_prompt)
        self.history.append({"role": "user", "content": user_prompt})

        final_answer = ""
        edited_files_since_verify: List[str] = []

        for step in range(1, self.config.max_agent_steps + 1):
            # Step 1: Compact context if approaching ceiling
            self.history, did_compact, before_tok, after_tok = compact_conversation(
                self.history,
                max_context_tokens=self.config.context_size,
                threshold_ratio=self.config.context_compact_threshold,
            )
            if did_compact:
                emit(
                    AgentEvent(
                        kind="compacted",
                        text=f"Contexto compactado automaticamente: {before_tok} -> {after_tok} tokens.",
                    )
                )

            payload: Dict[str, Any] = {
                "model": self.config.active_model_id,
                "messages": self.history,
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
                "temperature": 0.2,
                "top_p": 0.9,
                "presence_penalty": 0.15,  # Prevents low-bit MoE repetition loops!
            }

            try:
                resp = self._client.post(self.completions_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                err_msg = (
                    f"Falha de conexão com o motor local ({self.completions_url}): {exc}. "
                    "Certifique-se de que o motor está ativo (`nglai engine start`)."
                )
                emit(AgentEvent(kind="error", text=err_msg))
                return err_msg

            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            raw_content = str(message.get("content") or "")
            reasoning_field = str(message.get("reasoning_content") or "")
            api_tool_calls = message.get("tool_calls") or []

            inline_think, visible_text, xml_tool_calls = extract_thinking_and_fallback_tools(raw_content)
            combined_think = "\n\n".join(t for t in (reasoning_field.strip(), inline_think.strip()) if t)
            tool_calls = api_tool_calls if api_tool_calls else xml_tool_calls

            if combined_think:
                emit(AgentEvent(kind="thinking", text=combined_think, metadata={"step": step}))

            if not tool_calls:
                # Compound Self-Verification Check: if code was modified and never verified, prompt quick verification
                if self.config.auto_verify_edits and edited_files_since_verify and step < self.config.max_agent_steps - 1:
                    files_list = ", ".join(sorted(set(edited_files_since_verify)))
                    edited_files_since_verify.clear()
                    self.history.append({"role": "assistant", "content": visible_text})
                    self.history.append(
                        {
                            "role": "user",
                            "content": (
                                f"[AUTO-VERIFICADOR COMPOUND AI] Você modificou ({files_list}). "
                                "Se aplicável, execute um teste rápido ou validação via `run_command` para garantir "
                                "zero regressão antes de encerrar; caso já esteja 100% validado, entregue a resposta final."
                            ),
                        }
                    )
                    continue

                final_answer = visible_text
                self.history.append({"role": "assistant", "content": visible_text})
                if visible_text:
                    emit(AgentEvent(kind="content", text=visible_text))
                emit(
                    AgentEvent(
                        kind="done",
                        text=final_answer,
                        metadata={
                            "steps": step,
                            "context_tokens": estimate_history_tokens(self.history),
                        },
                    )
                )
                return final_answer

            # Record assistant message with normalized tool_calls
            if visible_text:
                emit(AgentEvent(kind="content", text=visible_text))

            self.history.append(
                {
                    "role": "assistant",
                    "content": visible_text or "",
                    "tool_calls": tool_calls,
                }
            )

            # Execute each requested tool sequentially with real-time events
            for tc in tool_calls:
                tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                fn_obj = tc.get("function") or {}
                fn_name = str(fn_obj.get("name") or "")
                raw_args = fn_obj.get("arguments") or "{}"

                try:
                    parsed_args = _repair_and_parse_json(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except Exception as exc:
                    result = ToolResult(
                        name=fn_name,
                        ok=False,
                        output=f"JSON de argumentos inválido ({exc}). Reenvie argumentos JSON válidos.",
                    )
                    parsed_args = {}
                else:
                    emit(AgentEvent(kind="tool_call", tool_name=fn_name, tool_args=parsed_args))
                    result = self.executor.execute(fn_name, parsed_args)

                if fn_name in ("write_file", "edit_file") and result.ok:
                    edited_files_since_verify.append(str(parsed_args.get("path", "")))
                elif fn_name == "run_command" and result.ok:
                    edited_files_since_verify.clear()

                emit(
                    AgentEvent(
                        kind="tool_result",
                        text=result.output,
                        tool_name=fn_name,
                        diff=result.diff,
                        metadata={"ok": result.ok},
                    )
                )

                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": fn_name,
                        "content": result.output,
                    }
                )

        timeout_msg = f"Limite máximo de passos autônomos ({self.config.max_agent_steps}) atingido."
        emit(AgentEvent(kind="done", text=timeout_msg))
        return final_answer or timeout_msg
