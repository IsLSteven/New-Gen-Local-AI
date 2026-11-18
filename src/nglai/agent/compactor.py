"""Smart Context Compactor (pairs with TurboQuant KV Cache for ultra-long agent sessions)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """Fast heuristic token estimator (~3.8 chars/token for code + PT/EN prose)."""
    chars = len(str(msg.get("content") or ""))
    if msg.get("tool_calls"):
        chars += len(str(msg["tool_calls"]))
    return max(4, int(chars / 3.8))


def estimate_history_tokens(messages: List[Dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


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

    compacted_notes: List[str] = []
    preserved_older: List[Dict[str, Any]] = []

    for msg in older:
        role = msg.get("role")
        content = str(msg.get("content") or "")
        if role == "user":
            preview = content.strip().replace("\n", " ")[:280]
            compacted_notes.append(f"- Pedido anterior do usuário: {preview}")
        elif role == "tool":
            name = msg.get("name") or "tool"
            first_line = content.splitlines()[0][:200] if content else ""
            compacted_notes.append(f"- Resultado [{name}]: {first_line}")
        elif role == "assistant":
            # Strip heavy <think> blocks from old turns to reclaim KV cache
            clean = content
            if "<think>" in clean and "</think>" in clean:
                _, _, after_think = clean.partition("</think>")
                clean = after_think.strip()
            if clean:
                compacted_notes.append(f"- Síntese do agente: {clean.replace(chr(10), ' ')[:260]}")

    summary_block = {
        "role": "user",
        "content": (
            "[HISTÓRICO ANTERIOR COMPACTADO AUTOMATICAMENTE PELO CONTEXT COMPACTOR]\n"
            + "\n".join(compacted_notes[-30:])
        ),
    }

    rebuilt = [*system_msgs, summary_block, *preserved_older, *recent]
    after = estimate_history_tokens(rebuilt)
    return rebuilt, True, before, after
