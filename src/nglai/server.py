"""FastAPI Backend: Antigravity-Style Web Workspace + Live Model Downloader + OpenAI /v1 Gateway."""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent.compactor import compact_openai_payload_messages
from .agent.orchestrator import AgentEvent, CompoundCodingAgent
from .agent.repomap import IGNORED_DIRS, RepoMapper
from .config import NGLAIConfig
from .engine.catalog import MODEL_CATALOG, get_model_spec
from .engine.hardware import calculate_surgical_split, detect_hardware
from .engine.installer import install_llama_engine
from .engine.runtime import EngineRuntime


class ModelSelectRequest(BaseModel):
    model_id: str
    context_size: Optional[int] = None
    enable_thinking: Optional[bool] = None


class ModelPullRequest(BaseModel):
    model_id: Optional[str] = None


class StoragePathsRequest(BaseModel):
    hot_ssd_dir: str
    cold_hdd_dir: str


_CTX_LIMIT_RE = re.compile(r"available context size \((\d+) tokens\)", re.IGNORECASE)


def _parse_upstream_error(raw_bytes: bytes, status_code: int) -> Dict[str, Any]:
    """Always normalize upstream HTTP error responses into valid OpenAI-style JSON."""
    raw_text = raw_bytes.decode("utf-8", errors="replace").strip()
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed if "error" in parsed else {"error": parsed}
    except Exception:
        pass
    return {
        "error": {
            "message": raw_text or f"Upstream engine returned HTTP {status_code}",
            "type": "upstream_error",
            "code": status_code,
        }
    }


def _extract_context_limit_from_error(err_payload: Dict[str, Any], fallback: int) -> Optional[int]:
    """Return context limit if upstream failed due to exceed_context_size_error, else None."""
    err_obj = err_payload.get("error") if isinstance(err_payload, dict) else None
    msg = str(err_obj.get("message") if isinstance(err_obj, dict) else err_payload)
    err_type = str(err_obj.get("type") if isinstance(err_obj, dict) else "")
    if "exceed_context_size_error" in err_type or "exceeds the available context size" in msg.lower():
        match = _CTX_LIMIT_RE.search(msg)
        return int(match.group(1)) if match else fallback
    return None


def create_app(
    config: Optional[NGLAIConfig] = None,
    workspace_root: Optional[Path] = None,
    runtime_instance: Optional[EngineRuntime] = None,
) -> FastAPI:
    cfg = config or NGLAIConfig.load()
    ws_root = (workspace_root or Path.cwd()).resolve()
    runtime = runtime_instance or EngineRuntime(cfg)

    download_state: Dict[str, Any] = {
        "active": False,
        "model_id": None,
        "downloaded_bytes": 0,
        "total_bytes": 0,
        "error": None,
        "done_path": None,
    }
    dl_lock = threading.Lock()

    app = FastAPI(
        title="New-Gen-Local-AI Workspace & Gateway",
        version="0.1.0",
        description="Triple-Stack Compressed Local Coding Agent + OpenAI-Compatible /v1 Gateway",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
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
        has_binary = runtime.find_llama_server_binary() is not None
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
                "binary_installed": has_binary,
                "endpoint": eng_status.endpoint,
                "pid": eng_status.pid,
                "turboquant_enabled": eng_status.turboquant_enabled,
                "kv_cache_type": eng_status.kv_cache_type,
                "message": eng_status.message,
            },
            "download": dict(download_state),
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

    @app.post("/api/vault/pull")
    async def pull_model_background(req: ModelPullRequest) -> Dict[str, Any]:
        target_id = req.model_id or cfg.active_model_id
        if target_id not in MODEL_CATALOG:
            raise HTTPException(status_code=404, detail=f"Modelo '{target_id}' inválido.")
        with dl_lock:
            if download_state["active"]:
                return {"ok": True, "already_running": True, "state": dict(download_state)}
            download_state.update(
                {
                    "active": True,
                    "model_id": target_id,
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "error": None,
                    "done_path": None,
                }
            )

        def _bg_download() -> None:
            def _prog(done: int, total: int) -> None:
                download_state["downloaded_bytes"] = done
                download_state["total_bytes"] = total

            try:
                path = runtime.vault.download_model(target_id, progress_cb=_prog, activate_hot=True)
                download_state["done_path"] = str(path)
            except Exception as exc:
                download_state["error"] = str(exc)
            finally:
                download_state["active"] = False

        threading.Thread(target=_bg_download, daemon=True).start()
        return {"ok": True, "model_id": target_id}

    @app.get("/api/vault/pull-status")
    async def get_pull_status() -> Dict[str, Any]:
        return dict(download_state)

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

    @app.post("/api/engine/install")
    async def install_engine_api() -> Dict[str, Any]:
        try:
            exe = await asyncio.to_thread(install_llama_engine, cfg, "auto", None)
            return {"ok": True, "binary": str(exe)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/engine/start")
    async def start_engine_api() -> Dict[str, Any]:
        try:
            if runtime.find_llama_server_binary() is None:
                await asyncio.to_thread(install_llama_engine, cfg, "auto", None)
            st = await asyncio.to_thread(runtime.start, cfg.active_model_id, 120)
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
        if not runtime.is_healthy():
            await asyncio.to_thread(runtime.start, cfg.active_model_id, 90)
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
        active_ctx = runtime._active_plan.recommended_ctx if runtime._active_plan else cfg.context_size

        client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=600.0))
        try:
            req = client.build_request("POST", target_url, json=body)
            upstream = await client.send(req, stream=is_stream)

            # Auto-recover seamlessly if client history + tool schemas overflowed local slot context
            if upstream.status_code == 400 and isinstance(body.get("messages"), list):
                err_bytes = await upstream.aread()
                await upstream.aclose()
                err_payload = _parse_upstream_error(err_bytes, upstream.status_code)
                ctx_limit = _extract_context_limit_from_error(err_payload, fallback=active_ctx)
                if ctx_limit is not None:
                    compacted_msgs, changed = compact_openai_payload_messages(
                        body["messages"],
                        max_context_tokens=ctx_limit,
                    )
                    if changed:
                        body["messages"] = compacted_msgs
                        retry_req = client.build_request("POST", target_url, json=body)
                        upstream = await client.send(retry_req, stream=is_stream)
                    else:
                        await client.aclose()
                        return JSONResponse(status_code=400, content=err_payload)
                else:
                    await client.aclose()
                    return JSONResponse(status_code=400, content=err_payload)

            if upstream.status_code >= 400:
                err_bytes = await upstream.aread()
                await upstream.aclose()
                await client.aclose()
                return JSONResponse(
                    status_code=upstream.status_code,
                    content=_parse_upstream_error(err_bytes, upstream.status_code),
                )

            if is_stream:
                async def event_stream() -> Any:
                    try:
                        async for chunk in upstream.aiter_bytes():
                            yield chunk
                    except Exception as exc:
                        err_json = json.dumps(
                            {
                                "error": {
                                    "message": f"Erro de conexão no stream com {target_url}: {exc}",
                                    "type": "proxy_stream_error",
                                }
                            },
                            ensure_ascii=False,
                        )
                        yield f"data: {err_json}\n\n".encode("utf-8")
                    finally:
                        await upstream.aclose()
                        await client.aclose()

                return StreamingResponse(event_stream(), media_type="text/event-stream")

            raw_ok = await upstream.aread()
            await upstream.aclose()
            await client.aclose()
            return JSONResponse(status_code=upstream.status_code, content=json.loads(raw_ok))
        except Exception as exc:
            await client.aclose()
            raise HTTPException(
                status_code=503,
                detail=f"Motor llama-server indisponível em {target_url}: {exc}",
            ) from exc

    return app
