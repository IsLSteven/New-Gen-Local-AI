"""Tests for Smart Context Compactor, <think>/XML Tool Recovery & Compound AI Orchestrator Loop."""

import json
from pathlib import Path
from typing import List

import httpx

from nglai.agent.compactor import compact_conversation, compact_openai_payload_messages
from nglai.agent.orchestrator import (
    AgentEvent,
    CompoundCodingAgent,
    extract_thinking_and_fallback_tools,
)
from nglai.config import NGLAIConfig


def test_extract_thinking_and_xml_tool_call_with_json_healing() -> None:
    raw = (
        "<think>Vou inspecionar o arquivo app.py primeiro.</think>\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "app.py",}}</tool_call>'
    )
    think, visible, calls = extract_thinking_and_fallback_tools(raw)
    assert "inspecionar o arquivo app.py" in think
    assert visible == ""
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "app.py"}


def test_context_compactor_shrinks_history() -> None:
    msgs = [{"role": "system", "content": "System rules"}]
    for i in range(25):
        msgs.append({"role": "user", "content": f"Tarefa {i}: " + ("x" * 600)})
        msgs.append({"role": "tool", "name": "read_file", "content": f"Arquivo {i}\n" + ("linha\n" * 200)})

    compacted, did_compact, before, after = compact_conversation(
        msgs, max_context_tokens=4000, threshold_ratio=0.5, keep_recent_messages=6
    )
    assert did_compact is True
    assert after < before
    assert compacted[0]["role"] == "system"


def test_openai_payload_compactor_preserves_atomic_tool_pairs() -> None:
    msgs = [{"role": "system", "content": "System rules"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"Pedido {i}: " + ("abc " * 400)})
        msgs.append(
            {
                "role": "assistant",
                "content": "<think>longo pensamento antigo</think>",
                "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "name": "read_file", "content": "ok " * 300})

    compacted, changed = compact_openai_payload_messages(msgs, max_context_tokens=8192)
    assert changed is True
    assert compacted[0]["role"] == "system"
    # No orphan 'tool' message immediately after system or summary
    for idx, m in enumerate(compacted):
        if m.get("role") == "tool":
            assert idx > 0
            prev = compacted[idx - 1]
            assert prev.get("role") in ("assistant", "tool")


def test_compound_agent_end_to_end_mocked_loop(tmp_path: Path) -> None:
    (tmp_path / "hello.py").write_text("def greet():\n    return 'hi'\n", encoding="utf-8")

    step_counter = {"n": 0}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        step_counter["n"] += 1
        n = step_counter["n"]
        if n == 1:
            # Step 1: Think + Read file via XML fallback tool call
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    "<think>Preciso ler hello.py antes de editar.</think>"
                                    '<tool_call>{"name": "read_file", "arguments": {"path": "hello.py"}}</tool_call>'
                                ),
                            }
                        }
                    ]
                },
            )
        if n == 2:
            # Step 2: Surgical edit via standard OpenAI tool_calls
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_edit_1",
                                        "type": "function",
                                        "function": {
                                            "name": "edit_file",
                                            "arguments": json.dumps(
                                                {
                                                    "path": "hello.py",
                                                    "target": "    return 'hi'",
                                                    "replacement": "    return 'olá mundo'",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        # Step 3+: Deliver final verified response
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Refatoração concluída e validada com sucesso em `hello.py`!",
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(mock_handler)
    client = httpx.Client(transport=transport)
    cfg = NGLAIConfig(base_dir=str(tmp_path), auto_verify_edits=False)
    agent = CompoundCodingAgent(config=cfg, workspace_root=tmp_path, http_client=client)

    events: List[AgentEvent] = []
    answer = agent.run_turn("Mude o retorno de greet() em hello.py para 'olá mundo'", on_event=events.append)

    assert "concluída" in answer
    assert "olá mundo" in (tmp_path / "hello.py").read_text(encoding="utf-8")
    kinds = [e.kind for e in events]
    assert "thinking" in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "done" in kinds
