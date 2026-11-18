"""FastAPI Backend: Antigravity-Style Web Workspace + Live Telemetry + OpenAI /v1 Gateway."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent.orchestrator import AgentEvent, CompoundCodingAgent
from .agent.repomap import IGNORED_DIRS, RepoMapper
from .config import NGLAIConfig
from .engine.catalog import MODEL_CATALOG, get_model_spec
from .engine.hardware import calculate_surgical_split, detect_hardware
from .engine.runtime import EngineRuntime


class ModelSelectRequest(BaseModel):
    model_id: str
    context_size: Optional[int] = None
    enable_thinking: Optional[bool] = None


class StoragePathsRequest(BaseModel):
    hot_ssd_dir: str
    cold_hdd_dir: str


def create_app(
    config: Optional[NGLAIConfig] = None,
    workspace_root: Optional[Path] = None,
) -> FastAPI:
    cfg = config or NGLAIConfig.load()
    ws_root = (workspace_root or Path.cwd()).resolve()
    runtime = EngineRuntime(cfg)

    app = FastAPI(
        title="New-Gen-Local-AI Workspace & Gateway",
        version="0.1.0",
        description="Triple-Stack Compressed Local Coding Agent + OpenAI-Compatible /v1 Gateway",
    )
    app.state.config = cfg
    app.state.workspace_root = ws_root
    app.state.runtime = runtime

    web_dir = Path(__file__).resolve().parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/")
    async def serve_index() -> Any:
        index_file = web_dir / "index.html"
        if not index_file.exists():
            return JSONResponse({"status": "ok", "service": "New-Gen-Local-AI"})
        return FileResponse(str(index_file))

    @app.get("/api/telemetry")
    async def get_telemetry() -> Dict[str, Any]:
        hw = detect_hardware(cfg.hot_ssd_dir, cfg.cold_hdd_dir)
        spec = get_model_spec(cfg.active_model_id) or next(iter(MODEL_CATALOG.values()))
        plan = calculate_surgical_split(hw, spec, requested_ctx=cfg.context_size, kv_mode=cfg.kv_cache_mode)
        eng_status = runtime.status()
        return {
            "workspace": str(ws_root),
            "hardware": hw.to_dict(),
            "active_model": {
                "id": spec.id,
                "name": spec.name,
                "size_gb": spec.size_gb,
                "total_params_b": spec.total_params_b,
                "active_params_b": spec.active_params_b,
                "quant_method": spec.quant_method,
            },
            "split_plan": {
                "feasible": plan.feasible,
                "execution_mode": plan.execution_mode,
                "n_gpu_layers": plan.n_gpu_layers,
                "cpu_moe_layers": plan.cpu_moe_layers,
                "override_tensor_regex": plan.override_tensor_regex,
                "recommended_ctx": plan.recommended_ctx,
                "kv_cache_type": plan.kv_cache_type_k,
                "estimated_vram_gb": plan.estimated_vram_gb,
                "estimated_ram_gb": plan.estimated_ram_gb,
                "estimated_tok_per_sec": plan.estimated_tok_per_sec,
                "notes_pt": plan.notes_pt,
            },
            "engine": {
                "running": eng_status.running,
                "endpoint": eng_status.endpoint,
                "pid": eng_status.pid,
                "turboquant_enabled": eng_status.turboquant_enabled,
                "kv_cache_type": eng_status.kv_cache_type,
                "message": eng_status.message,
            },
            "config": {
                "hot_ssd_dir": cfg.hot_ssd_dir,
                "cold_hdd_dir": cfg.cold_hdd_dir,
                "context_size": cfg.context_size,
                "enable_thinking": cfg.enable_thinking,
            },
        }

    @app.get("/api/models")
    async def get_models_catalog() -> Dict[str, Any]:
        vault_entries = {e.model_id: e for e in runtime.vault.inspect_catalog()}
        items: List[Dict[str, Any]] = []
        for spec in MODEL_CATALOG.values():
            ve = vault_entries.get(spec.id)
            items.append(
                {
                    "id": spec.id,
                    "name": spec.name,
                    "size_gb": spec.size_gb,
                    "total_params_b": spec.total_params_b,
                    "active_params_b": spec.active_params_b,
                    "is_moe": spec.is_moe,
                    "quant_method": spec.quant_method,
                    "tier": spec.tier,
                    "supports_thinking": spec.supports_thinking,
                    "description_pt": spec.description_pt,
                    "in_hot_ssd": ve.in_hot_ssd if ve else False,
                    "in_cold_hdd": ve.in_cold_hdd if ve else False,
                    "active": spec.id == cfg.active_model_id,
                }
            )
        return {"models": items, "active_model_id": cfg.active_model_id}

    @app.post("/api/models/select")
    async def select_model(req: ModelSelectRequest) -> Dict[str, Any]:
        if req.model_id not in MODEL_CATALOG:
            raise HTTPException(status_code=404, detail=f"Modelo '{req.model_id}' não existe no catálogo.")
        cfg.active_model_id = req.model_id
        if req.context_size is not None:
            cfg.context_size = max(4096, min(262144, req.context_size))
        if req.enable_thinking is not None:
            cfg.enable_thinking = req.enable_thinking
        cfg.save()
        return {"ok": True, "active_model_id": cfg.active_model_id, "context_size": cfg.context_size}

    @app.post("/api/vault/paths")
    async def update_vault_paths(req: StoragePathsRequest) -> Dict[str, Any]:
        cfg.hot_ssd_dir = str(Path(req.hot_ssd_dir).resolve())
        cfg.cold_hdd_dir = str(Path(req.cold_hdd_dir).resolve())
        cfg.ensure_directories()
        cfg.save()
        runtime.vault = EngineRuntime(cfg).vault
        return {"ok": True, "hot_ssd_dir": cfg.hot_ssd_dir, "cold_hdd_dir": cfg.cold_hdd_dir}

    @app.post("/api/vault/evict")
    async def evict_hot_ssd() -> Dict[str, Any]:
        evicted = runtime.vault.evict_inactive_from_hot(keep_model_id=cfg.active_model_id)
        return {"ok": True, "evicted_files": evicted}

    @app.post("/api/engine/start")
    async def start_engine_api() -> Dict[str, Any]:
        try:
            st = await asyncio.to_thread(runtime.start, cfg.active_model_id, 60)
            return {"ok": st.running, "status": st.__dict__}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/engine/stop")
    async def stop_engine_api() -> Dict[str, Any]:
        await asyncio.to_thread(runtime.stop)
        return {"ok": True}

    @app.get("/api/workspace/tree")
    async def list_workspace_tree() -> Dict[str, Any]:
        files: List[Dict[str, Any]] = []
        for dirpath, dirnames, filenames in os.walk(ws_root):
            dirnames[:] = [d for d in sorted(dirnames) if d not in IGNORED_DIRS and not d.startswith(".git")]
            for fname in sorted(filenames):
                full = Path(dirpath) / fname
                rel = full.relative_to(ws_root).as_posix()
                try:
                    size = full.stat().st_size
                except OSError:
                    size = 0
                files.append({"path": rel, "size": size})
                if len(files) >= 400:
                    break
        return {"root": str(ws_root), "files": files}

    @app.get("/api/workspace/file")
    async def read_workspace_file(path: str) -> Dict[str, Any]:
        candidate = (ws_root / path).resolve()
        try:
            candidate.relative_to(ws_root)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Acesso fora do workspace negado.") from exc
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Arquivo não encontrado.")
        content = candidate.read_text(encoding="utf-8", errors="replace")
        return {"path": path, "content": content[:200_000]}

    @app.get("/api/workspace/repomap")
    async def get_workspace_repomap(query: str = "") -> Dict[str, Any]:
        mapper = RepoMapper(ws_root)
        text = await asyncio.to_thread(mapper.build_map_text, query, cfg.repomap_max_tokens)
        return {"repomap": text}

    # =========================================================================
    # WebSocket Autonomous Agent Stream (Antigravity-Style Experience)
    # =========================================================================
    @app.websocket("/ws/agent")
    async def websocket_agent(ws: WebSocket) -> None:
        await ws.accept()
        agent = CompoundCodingAgent(config=cfg, workspace_root=ws_root)
        loop = asyncio.get_running_loop()

        try:
            while True:
                msg = await ws.receive_json()
                action = msg.get("action", "chat")
                if action == "reset":
                    agent.reset_session()
                    await ws.send_json({"kind": "reset_ok"})
                    continue

                prompt = str(msg.get("prompt") or "").strip()
                if not prompt:
                    continue

                if "enable_thinking" in msg:
                    cfg.enable_thinking = bool(msg["enable_thinking"])

                queue: asyncio.Queue[Optional[AgentEvent]] = asyncio.Queue()

                def on_event(ev: AgentEvent) -> None:
                    loop.call_soon_threadsafe(queue.put_nowait, ev)

                async def run_worker() -> None:
                    try:
                        await asyncio.to_thread(agent.run_turn, prompt, on_event)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                worker_task = asyncio.create_task(run_worker())

                while True:
                    ev = await queue.get()
                    if ev is None:
                        break
                    await ws.send_json(
                        {
                            "kind": ev.kind,
                            "text": ev.text,
                            "tool_name": ev.tool_name,
                            "tool_args": ev.tool_args,
                            "diff": ev.diff,
                            "metadata": ev.metadata,
                        }
                    )

                await worker_task
        except WebSocketDisconnect:
            return

    # =========================================================================
    # OpenAI-Compatible /v1 Gateway (VS Code Cline / Roo Code / Continue / OpenCode)
    # =========================================================================
    @app.get("/v1/models")
    async def openai_list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": spec.id,
                    "object": "model",
                    "owned_by": "new-gen-local-ai",
                }
                for spec in MODEL_CATALOG.values()
            ],
        }

    @app.post("/v1/chat/completions")
    async def openai_chat_completions_proxy(request: Request) -> Any:
        body = await request.json()
        # Optional Repo-Map enrichment for external IDE extensions
        if request.headers.get("X-NGLAI-RepoMap", "").lower() in ("1", "true", "yes"):
            msgs = body.get("messages") or []
            last_user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
            rmap = RepoMapper(ws_root).build_map_text(query=str(last_user), max_tokens=1500)
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["content"] = str(msgs[0].get("content", "")) + "\n\n" + rmap
            else:
                msgs.insert(0, {"role": "system", "content": rmap})
            body["messages"] = msgs

        target_url = f"{runtime.base_url}/v1/chat/completions"
        is_stream = bool(body.get("stream", False))

        if is_stream:
            async def event_Connect() -> Any:
                try:
                    async with httpx.AsyncClient(timeout=600.0) as client:
                        async with client.stream("POST", target_url, json=body) as upstream:
                            upstream.raise_for_status()
                            async for chunk in upstream.aiter_bytes():
                                yield chunk
                except Exception as exc:
                    err = f'data: {{"error": "Engine local offline em {target_url}: {exc}"}}\n\n'
                    yield err.encode("utf-8")

            return StreamingResponse(event_Connect(), media_type="text/event-stream")

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(target_url, json=body)
                return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Motor llama-server indisponível em {target_url}: {exc}",
            ) from exc

    return app
