"""Native Model Context Protocol (MCP) Stdio Server for Google Antigravity & Claude/Cursor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from .agent.orchestrator import CompoundCodingAgent
from .config import NGLAIConfig
from .engine.catalog import MODEL_CATALOG, get_model_spec
from .engine.hardware import calculate_surgical_split, detect_hardware
from .engine.runtime import EngineRuntime

MCP_TOOLS = [
    {
        "name": "ask_local_ai",
        "description": (
            "Envia uma pergunta ou tarefa de geração/revisão de código diretamente para o LLM local "
            "comprimido do New-Gen-Local-AI (zero custo de nuvem e 100% privado)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Instrução ou código para a IA local analisar/gerar."},
                "max_tokens": {"type": "integer", "description": "Máximo de tokens na resposta (padrão 1024)."},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "run_local_coding_agent",
        "description": (
            "Aciona o Agente Autônomo Compound AI local (com AST Repo-Map, leitura/edição de arquivos e "
            "auto-verificação sintática) para executar uma tarefa em um workspace."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Descrição detalhada da tarefa de engenharia."},
                "workspace": {"type": "string", "description": "Caminho do diretório do projeto (opcional)."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "get_local_ai_status",
        "description": "Retorna diagnóstico de GPU/VRAM/RAM, modelo ativo e status do motor New-Gen-Local-AI.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def handle_mcp_request(req: Dict[str, Any], cfg: Optional[NGLAIConfig] = None) -> Optional[Dict[str, Any]]:
    """Process a single JSON-RPC 2.0 MCP message and return response dict (or None for notifications)."""
    method = req.get("method", "")
    msg_id = req.get("id")
    params = req.get("params") or {}

    if msg_id is None:
        # Notification (e.g. notifications/initialized)
        return None

    config = cfg or NGLAIConfig.load()

    if method == "initialize":
        proto = params.get("protocolVersion") or "2024-11-05"
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": proto,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "new-gen-local-ai", "version": "0.1.0"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": MCP_TOOLS}}

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        try:
            text_out = _dispatch_mcp_tool(name, args, config)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": [{"type": "text", "text": text_out}], "isError": False},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Erro na ferramenta local {name}: {exc}"}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _dispatch_mcp_tool(name: str, args: Dict[str, Any], cfg: NGLAIConfig) -> str:
    runtime = EngineRuntime(cfg)

    if name == "get_local_ai_status":
        hw = detect_hardware(cfg.hot_ssd_dir, cfg.cold_hdd_dir)
        spec = get_model_spec(cfg.active_model_id) or next(iter(MODEL_CATALOG.values()))
        plan = calculate_surgical_split(hw, spec, requested_ctx=cfg.context_size, kv_mode=cfg.kv_cache_mode)
        st = runtime.status()
        return json.dumps(
            {
                "engine_running": st.running,
                "endpoint": st.endpoint,
                "active_model": spec.name,
                "execution_mode": plan.execution_mode,
                "estimated_vram_gb": plan.estimated_vram_gb,
                "estimated_ram_gb": plan.estimated_ram_gb,
                "gpu": hw.primary_gpu.name if hw.primary_gpu else "CPU",
            },
            indent=2,
            ensure_ascii=False,
        )

    if name == "ask_local_ai":
        if not runtime.is_healthy():
            runtime.start(cfg.active_model_id, wait_seconds=90)
        prompt = str(args.get("prompt") or "")
        max_tokens = int(args.get("max_tokens") or 1024)
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{runtime.base_url}/v1/chat/completions",
                json={
                    "model": cfg.active_model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    if name == "run_local_coding_agent":
        if not runtime.is_healthy():
            runtime.start(cfg.active_model_id, wait_seconds=90)
        ws = Path(args["workspace"]).resolve() if args.get("workspace") else Path.cwd().resolve()
        agent = CompoundCodingAgent(config=cfg, workspace_root=ws)
        return agent.run_turn(str(args.get("task") or ""))

    raise ValueError(f"Ferramenta MCP desconhecida: {name}")


def run_stdio_mcp_server() -> int:
    """Run newline-delimited JSON-RPC 2.0 MCP loop over stdin/stdout."""
    cfg = NGLAIConfig.load()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_mcp_request(req, cfg=cfg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0
