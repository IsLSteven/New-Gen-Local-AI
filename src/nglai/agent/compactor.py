"""Smart Context Compactor (pairs with TurboQuant KV Cache and /v1 Gateway auto-recovery)."""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _content_to_text(content: Any) -> str:
    """Extract plain text from OpenAI string or multimodal list content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _strip_think_blocks(text: str) -> str:
    if "<think>" not in text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].strip()
    return cleaned


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """Fast heuristic token estimator (~3.5 chars/token for code, schemas, and PT/EN prose)."""
    chars = len(_content_to_text(msg.get("content")))
    if msg.get("reasoning_content"):
        chars += len(str(msg["reasoning_content"]))
    if msg.get("tool_calls"):
        chars += len(str(msg["tool_calls"]))
    return max(6, int(chars / 3.5))


def estimate_history_tokens(messages: List[Dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def _group_atomic_turns(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group assistant tool_calls together with their subsequent tool responses so pairs never break."""
    groups: List[List[Dict[str, Any]]] = []
    idx = 0
    while idx < len(messages):
        msg = messages[idx]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            group = [msg]
            idx += 1
            while idx < len(messages) and messages[idx].get("role") == "tool":
                group.append(messages[idx])
                idx += 1
            groups.append(group)
        else:
            groups.append([msg])
            idx += 1
    return groups


def _summarize_messages(older: List[Dict[str, Any]]) -> Dict[str, Any]:
    notes: List[str] = []
    for msg in older:
        role = msg.get("role")
        text = _strip_think_blocks(_content_to_text(msg.get("content")))
        if role == "user":
            preview = text.strip().replace("\n", " ")[:300]
            if preview:
                notes.append(f"- Pedido anterior do usuário: {preview}")
        elif role == "tool":
            name = msg.get("name") or "tool"
            first_line = text.splitlines()[0][:200] if text else ""
            notes.append(f"- Resultado [{name}]: {first_line}")
        elif role == "assistant":
            if msg.get("tool_calls") and not text:
                tc_names = [
                    str((tc.get("function") or {}).get("name") or "tool")
                    for tc in (msg.get("tool_calls") or [])
                ]
                notes.append(f"- Ação anterior do agente: {', '.join(tc_names)}")
            elif text:
                notes.append(f"- Síntese do agente: {text.replace(chr(10), ' ')[:260]}")

    return {
        "role": "user",
        "content": (
            "[HISTÓRICO ANTERIOR COMPACTADO AUTOMATICAMENTE PELO CONTEXT COMPACTOR]\n"
            + ("\n".join(notes[-30:]) if notes else "- Turnos anteriores resumidos para caber na janela local.")
        ),
    }


def compact_conversation(
    messages: List[Dict[str, Any]],
    max_context_tokens: int = 32768,
    threshold_ratio: float = 0.78,
    keep_recent_messages: int = 10,
) -> Tuple[List[Dict[str, Any]], bool, int, int]:
    """
    Compress older verbose tool outputs and intermediate steps when approaching context ceiling,
    preserving system prompt, user goals, file edit records, and recent turns intact.
    Returns (new_messages, did_compact, before_tokens, after_tokens).
    """
    before = estimate_history_tokens(messages)
    budget = int(max_context_tokens * threshold_ratio)
    if before <= budget or len(messages) <= (keep_recent_messages + 2):
        return messages, False, before, before

    system_msgs = [m for m in messages[:1] if m.get("role") == "system"]
    start_idx = 1 if system_msgs else 0
    cutoff = max(start_idx, len(messages) - keep_recent_messages)

    older = messages[start_idx:cutoff]
    recent = messages[cutoff:]

    summary_block = _summarize_messages(older)
    rebuilt = [*system_msgs, summary_block, *recent]
    after = estimate_history_tokens(rebuilt)
    return rebuilt, True, before, after


def compact_openai_payload_messages(
    messages: List[Dict[str, Any]],
    max_context_tokens: int,
    tools_token_reserve: int = 1800,
    reply_reserve: int = 1024,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Safely compact an arbitrary OpenAI-format message list (e.g. from Cline / Roo Code)
    so it fits inside `max_context_tokens` while keeping tool_call/tool pairs valid.
    """
    if not messages:
        return messages, False

    working = copy.deepcopy(messages)
    changed = False

    # Pass 1: Strip <think> blocks and reasoning_content from all non-final assistant messages
    for idx in range(len(working) - 1):
        msg = working[idx]
        if msg.get("role") == "assistant":
            if msg.pop("reasoning_content", None) is not None:
                changed = True
            if isinstance(msg.get("content"), str) and "<think>" in msg["content"]:
                stripped = _strip_think_blocks(msg["content"])
                if stripped != msg["content"]:
                    msg["content"] = stripped
                    changed = True

    budget = max(1200, max_context_tokens - tools_token_reserve - reply_reserve)
    if estimate_history_tokens(working) <= budget:
        return working, changed

    # Separate leading system/developer instructions from conversational turns
    lead_end = 0
    while lead_end < len(working) and working[lead_end].get("role") in ("system", "developer"):
        lead_end += 1
    leading = working[:lead_end]
    body_groups = _group_atomic_turns(working[lead_end:])

    # Pass 2: Progressively collapse oldest atomic turn groups while keeping recent context
    keep_groups = max(1, min(len(body_groups), 6))
    while keep_groups >= 1:
        split_at = max(0, len(body_groups) - keep_groups)
        older_flat = [m for g in body_groups[:split_at] for m in g]
        recent_flat = [m for g in body_groups[split_at:] for m in g]

        candidate = [*leading]
        if older_flat:
            candidate.append(_summarize_messages(older_flat))
        candidate.extend(recent_flat)

        if estimate_history_tokens(candidate) <= budget or keep_groups == 1:
            working = candidate
            changed = True
            break
        keep_groups -= 1

    # Pass 3: If a single tool/user block is still enormous, trim oversized tool outputs softly
    if estimate_history_tokens(working) > budget:
        max_chars = max(1200, int((budget * 3.0) / max(1, len(working))))
        for msg in working[lead_end:-1]:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > max_chars:
                msg["content"] = msg["content"][:max_chars] + "\n... [conteúdo antigo truncado para caber no contexto]"
                changed = True

    return working, changed
