"""Rich Interactive CLI & Command Suite for New-Gen-Local-AI (nglai)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .agent.orchestrator import AgentEvent, CompoundCodingAgent
from .agent.repomap import RepoMapper
from .config import NGLAIConfig
from .engine.catalog import MODEL_CATALOG, get_model_spec
from .engine.hardware import calculate_surgical_split, detect_hardware
from .engine.installer import install_llama_engine
from .engine.runtime import EngineRuntime
from .server import create_app

console = Console()


def render_hardware_and_split(cfg: NGLAIConfig) -> None:
    hw = detect_hardware(cfg.hot_ssd_dir, cfg.cold_hdd_dir)
    spec = get_model_spec(cfg.active_model_id) or next(iter(MODEL_CATALOG.values()))
    plan = calculate_surgical_split(hw, spec, requested_ctx=cfg.context_size, kv_mode=cfg.kv_cache_mode)

    table = Table(title="⚡ Diagnóstico de Hardware & Tensor Split Cirúrgico", show_lines=True)
    table.add_column("Componente", style="bold cyan")
    table.add_column("Detectado", style="white")
    table.add_column("Alocação Triple-Stack", style="bold green")

    gpu_label = (
        f"{hw.primary_gpu.name} ({hw.primary_gpu.vram_total_gb} GB VRAM | Driver {hw.primary_gpu.driver_version})"
        if hw.primary_gpu
        else "Nenhuma GPU NVIDIA detectada (Modo CPU)"
    )
    table.add_row("CPU", f"{hw.cpu_name} ({hw.cpu_physical_cores}C / {hw.cpu_logical_threads}T)", f"{plan.cpu_threads} threads ativas")
    table.add_row("GPU (VRAM)", gpu_label, f"~{plan.estimated_vram_gb} GB alocados (-ngl {plan.n_gpu_layers})")
    table.add_row(
        "Memória RAM",
        f"{hw.ram_total_gb} GB Totais ({hw.ram_available_gb} GB livres)",
        f"~{plan.estimated_ram_gb} GB p/ {plan.cpu_moe_layers} blocos MoE em RAM",
    )
    if hw.hot_ssd_disk:
        table.add_row("Hot Cache (SSD)", f"{hw.hot_ssd_disk.path}", f"{hw.hot_ssd_disk.free_gb} GB livres")
    if hw.cold_hdd_disk:
        table.add_row("Cold Vault (HDD)", f"{hw.cold_hdd_disk.path}", f"{hw.cold_hdd_disk.free_gb} GB livres")
    table.add_row(
        "Modelo Ativo",
        f"{spec.name} ({spec.size_gb} GB)",
        f"Modo: {plan.execution_mode} | KV: {plan.kv_cache_type_k} ({plan.recommended_ctx} ctx) | {plan.estimated_tok_per_sec}",
    )

    console.print(table)
    for note in plan.notes_pt:
        console.print(f"  [dim]•[/dim] {note}")
    args_preview = " ".join(["llama-server", "-m", f'"{spec.filename}"', *plan.to_llama_args(turboquant_supported=True)])
    console.print(Panel(args_preview, title="Comando Otimizado Gerado (TurboQuant + Hybrid MoE)", border_style="blue"))


def render_catalog_table(cfg: NGLAIConfig, runtime: EngineRuntime) -> None:
    entries = {e.model_id: e for e in runtime.vault.inspect_catalog()}
    table = Table(title="📦 Catálogo Oficial Triple-Stack (Unsloth Dynamic 2.0 + MoE)", show_lines=True)
    table.add_column("ID", style="bold yellow")
    table.add_column("Modelo", style="white")
    table.add_column("Tam.", justify="right")
    table.add_column("Params (Ativos)", justify="center")
    table.add_column("Status Vault", justify="center")

    for spec in MODEL_CATALOG.values():
        ve = entries.get(spec.id)
        active_mark = "⭐ " if spec.id == cfg.active_model_id else ""
        if ve and ve.in_hot_ssd:
            vault_status = "[bold green]🔥 Hot SSD[/bold green]"
        elif ve and ve.in_cold_hdd:
            vault_status = "[cyan]❄️ Cold HDD[/cyan]"
        else:
            vault_status = "[dim]Não baixado[/dim]"

        table.add_row(
            f"{active_mark}{spec.id}",
            f"{spec.name}\n[dim]{spec.description_pt}[/dim]",
            f"{spec.size_gb} GB",
            f"{spec.total_params_b}B ({spec.active_params_b}B)",
            vault_status,
        )
    console.print(table)


def print_ide_integration(cfg: NGLAIConfig) -> None:
    gateway_url = f"http://{cfg.web_host}:{cfg.web_port}/v1"
    ide_cfg = {
        "provider": "OpenAI Compatible (New-Gen-Local-AI)",
        "baseUrl": gateway_url,
        "apiKey": "nglai-local",
        "modelId": cfg.active_model_id,
        "customHeaders": {"X-NGLAI-RepoMap": "true"},
    }
    console.print(
        Panel(
            Syntax(json.dumps(ide_cfg, indent=2, ensure_ascii=False), "json", theme="monokai"),
            title="🔌 Configuração Pronta p/ VS Code (Roo Code / Cline / Continue / OpenCode)",
            border_style="green",
        )
    )


def interactive_chat(cfg: NGLAIConfig, workspace: Path) -> None:
    runtime = EngineRuntime(cfg)

    def confirm_cmd(command: str, dangerous: bool) -> bool:
        tag = "[bold red]⚠️ COMANDO SENSÍVEL[/bold red]" if dangerous else "[yellow]Executar comando[/yellow]"
        console.print(f"\n{tag}: [bold white]{command}[/bold white]")
        ans = console.input("Permitir execução? [s/N]: ").strip().lower()
        return ans in ("s", "sim", "y", "yes")

    agent = CompoundCodingAgent(config=cfg, workspace_root=workspace, command_approval_cb=confirm_cmd)

    console.print(
        Panel.fit(
            f"[bold cyan]🚀 New-Gen-Local-AI[/bold cyan] — Agente Autônomo Compound AI\n"
            f"Workspace: [bold]{workspace}[/bold]\n"
            f"Modelo Ativo: [bold yellow]{cfg.active_model_id}[/bold yellow] | Thinking: [green]{cfg.enable_thinking}[/green]\n"
            f"Comandos: [dim]/hardware  /models  /use <id>  /map  /memory <nota>  /think  /ide  /clear  /sair[/dim]",
            border_style="cyan",
        )
    )

    hist_file = Path(cfg.cache_dir) / "cli_history.txt"
    session: PromptSession = PromptSession(history=FileHistory(str(hist_file)))

    def handle_event(ev: AgentEvent) -> None:
        if ev.kind == "thinking" and ev.text:
            console.print(Panel(ev.text, title="🧠 Deep Thinking (<think>)", border_style="magenta", style="dim"))
        elif ev.kind == "tool_call":
            args_short = json.dumps(ev.tool_args or {}, ensure_ascii=False)
            if len(args_short) > 160:
                args_short = args_short[:157] + "..."
            console.print(f"  [bold blue]🔧 {ev.tool_name}[/bold blue]([dim]{args_short}[/dim])")
        elif ev.kind == "tool_result":
            icon = "✅" if (ev.metadata or {}).get("ok") else "❌"
            first_line = (ev.text.splitlines() or [""])[0][:140]
            console.print(f"     {icon} [dim]{first_line}[/dim]")
            if ev.diff:
                console.print(Syntax(ev.diff[:3000], "diff", theme="monokai"))
        elif ev.kind == "compacted":
            console.print(f"  [yellow]🗜️ {ev.text}[/yellow]")
        elif ev.kind == "content" and ev.text:
            console.print(Panel(Markdown(ev.text), title="🤖 New-Gen-Local-AI", border_style="green"))
        elif ev.kind == "error":
            console.print(Panel(ev.text, title="Erro de Motor", border_style="red"))

    while True:
        try:
            user_input = session.prompt("\nnglai ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Até logo![/dim]")
            break

        if not user_input:
            continue
        if user_input in ("/sair", "/exit", "/quit"):
            break
        if user_input == "/help":
            console.print(
                "[cyan]/hardware[/cyan] diagnóstico GPU/RAM | [cyan]/models[/cyan] catálogo | "
                "[cyan]/use <id>[/cyan] trocar modelo | [cyan]/map[/cyan] AST Repo-Map | "
                "[cyan]/memory <texto>[/cyan] salvar regra | [cyan]/think[/cyan] alternar raciocínio | "
                "[cyan]/ide[/cyan] config VS Code | [cyan]/clear[/cyan] limpar sessão"
            )
            continue
        if user_input == "/hardware":
            render_hardware_and_split(cfg)
            continue
        if user_input == "/models":
            render_catalog_table(cfg, runtime)
            continue
        if user_input.startswith("/use "):
            new_id = user_input.split(maxsplit=1)[1].strip()
            if new_id in MODEL_CATALOG:
                cfg.active_model_id = new_id
                cfg.save()
                console.print(f"[green]✓ Modelo ativo alterado para: {new_id}[/green]")
            else:
                console.print(f"[red]Modelo '{new_id}' inválido. Use /models para listar.[/red]")
            continue
        if user_input == "/map":
            console.print(Panel(RepoMapper(workspace).build_map_text(max_tokens=cfg.repomap_max_tokens), title="🗺️ AST Repo-Map"))
            continue
        if user_input.startswith("/memory"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                agent.memory.add_note(parts[1].strip())
                console.print("[green]✓ Nota arquitetural salva na memória persistente do projeto.[/green]")
            else:
                console.print(agent.memory.render_for_prompt() or "[dim]Memória vazia.[/dim]")
            continue
        if user_input == "/think":
            cfg.enable_thinking = not cfg.enable_thinking
            cfg.save()
            console.print(f"[cyan]Modo Deep Thinking: {'ATIVADO' if cfg.enable_thinking else 'DESATIVADO'}[/cyan]")
            continue
        if user_input == "/ide":
            print_ide_integration(cfg)
            continue
        if user_input == "/clear":
            agent.reset_session()
            console.print("[dim]Sessão reiniciada.[/dim]")
            continue

        agent.run_turn(user_input, on_event=handle_event)


def main(argv: Optional[list[str]] = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", "-w", default=None, help="Diretório do projeto alvo")

    parser = argparse.ArgumentParser(
        prog="nglai",
        parents=[common],
        description="New-Gen-Local-AI: Agente Autônomo 30B-35B MoE + Compressão Triple-Stack",
    )
    sub = parser.add_subparsers(dest="subcommand")

    sub.add_parser("chat", parents=[common], help="Inicia o agente interativo no terminal (padrão)")
    sub.add_parser("hardware", parents=[common], help="Mostra diagnóstico de GPU/VRAM/RAM e cálculo de Tensor Split")
    sub.add_parser("ide", parents=[common], help="Exibe configuração OpenAI-compatible para VS Code (Roo Code / Cline)")

    web_p = sub.add_parser("web", parents=[common], help="Inicia o Painel Web estilo Antigravity + Gateway /v1")
    web_p.add_argument("--host", default=None)
    web_p.add_argument("--port", type=int, default=None)

    up_p = sub.add_parser("up", parents=[common], help="Tudo-em-um: instala motor se faltar, sobe modelo se baixado e abre o Painel Web")
    up_p.add_argument("--host", default=None)
    up_p.add_argument("--port", type=int, default=None)
    up_p.add_argument("--pull", action="store_true", help="Baixa automaticamente o modelo ativo se ainda não existir")

    vault_p = sub.add_parser("vault", help="Gerencia modelos entre Hot SSD (256GB) e Cold HDD (1TB)")
    vault_p.add_argument("action", choices=["list", "pull", "evict", "set-paths"])
    vault_p.add_argument("model_id", nargs="?", default=None)
    vault_p.add_argument("--hot-ssd", default=None, help="Pasta rápida no SSD")
    vault_p.add_argument("--cold-hdd", default=None, help="Cofre de modelos no HDD de 1TB")

    eng_p = sub.add_parser("engine", help="Controla o motor local llama-server + TurboQuant")
    eng_p.add_argument("action", choices=["status", "start", "stop", "cmd", "install"])
    eng_p.add_argument("--backend", default="auto", choices=["auto", "cuda", "vulkan", "cpu"])

    args = parser.parse_args(argv)
    cfg = NGLAIConfig.load()
    workspace = Path(args.workspace or ".").resolve()
    runtime = EngineRuntime(cfg)

    cmd = args.subcommand or "chat"
    if cmd == "hardware":
        render_hardware_and_split(cfg)
        return 0
    if cmd == "ide":
        print_ide_integration(cfg)
        return 0
    if cmd == "up":
        host = args.host or cfg.web_host
        port = args.port or cfg.web_port
        if runtime.find_llama_server_binary() is None:
            console.print("[cyan]⚙️ Binário llama-server não detectado. Instalando automaticamente p/ seu hardware...[/cyan]")

            def _inst_prog(fname: str, done: int, total: int) -> None:
                if total > 0:
                    sys.stdout.write(f"\r  ⬇ {fname}: {done / (1024**2):.1f}/{total / (1024**2):.1f} MB")
                    sys.stdout.flush()

            exe = install_llama_engine(cfg, backend="auto", progress_cb=_inst_prog)
            console.print(f"\n[green]✓ Motor instalado em: {exe}[/green]")

        if args.pull and runtime.vault.resolve_active_path(cfg.active_model_id, promote_to_hot=False) is None:
            console.print(f"[cyan]⬇ Baixando modelo ativo '{cfg.active_model_id}'...[/cyan]")

            def _pull_prog(done: int, total: int) -> None:
                if total > 0:
                    sys.stdout.write(f"\r  Progresso: {done / (1024**3):.2f}/{total / (1024**3):.2f} GB ({(done/total)*100:.1f}%)")
                    sys.stdout.flush()

            runtime.vault.download_model(cfg.active_model_id, progress_cb=_pull_prog, activate_hot=True)
            console.print("\n[green]✓ Modelo baixado![/green]")

        if runtime.vault.resolve_active_path(cfg.active_model_id, promote_to_hot=True) is not None:
            console.print(f"[cyan]🚀 Subindo motor local com '{cfg.active_model_id}'...[/cyan]")
            st = runtime.start(cfg.active_model_id, wait_seconds=90)
            console.print(f"[green]✓ {st.message} ({st.endpoint})[/green]")
        else:
            console.print(
                f"[yellow]ℹ️ Modelo '{cfg.active_model_id}' ainda não baixado. "
                f"Você pode baixá-lo com 1 clique direto no Painel Web![/yellow]"
            )

        console.print(
            Panel.fit(
                f"🌐 [bold green]New-Gen-Local-AI Pronto![/bold green]\n"
                f"Abra no navegador: [underline cyan]http://{host}:{port}[/underline cyan]\n"
                f"Gateway VS Code (Cline/Roo): [underline yellow]http://{host}:{port}/v1[/underline yellow]",
                border_style="green",
            )
        )
        app = create_app(config=cfg, workspace_root=workspace, runtime_instance=runtime)
        uvicorn.run(app, host=host, port=port, log_level="info")
        return 0
    if cmd == "web":
        host = args.host or cfg.web_host
        port = args.port or cfg.web_port
        console.print(
            Panel.fit(
                f"🌐 [bold green]New-Gen-Local-AI Web Workspace & Gateway /v1[/bold green]\n"
                f"Painel Web (Antigravity-Style): [underline cyan]http://{host}:{port}[/underline cyan]\n"
                f"OpenAI Gateway (VS Code Cline/Roo): [underline yellow]http://{host}:{port}/v1[/underline yellow]",
                border_style="green",
            )
        )
        app = create_app(config=cfg, workspace_root=workspace, runtime_instance=runtime)
        uvicorn.run(app, host=host, port=port, log_level="info")
        return 0
    if cmd == "vault":
        if args.action == "list":
            render_catalog_table(cfg, runtime)
        elif args.action == "evict":
            removed = runtime.vault.evict_inactive_from_hot(keep_model_id=cfg.active_model_id)
            console.print(f"[green]✓ Arquivos movidos do SSD para o Cold HDD: {removed or 'nenhum inativo'}[/green]")
        elif args.action == "set-paths":
            if args.hot_ssd:
                cfg.hot_ssd_dir = str(Path(args.hot_ssd).resolve())
            if args.cold_hdd:
                cfg.cold_hdd_dir = str(Path(args.cold_hdd).resolve())
            cfg.ensure_directories()
            cfg.save()
            console.print(f"[green]✓ Hot SSD: {cfg.hot_ssd_dir}\n✓ Cold HDD: {cfg.cold_hdd_dir}[/green]")
        elif args.action == "pull":
            target = args.model_id or cfg.active_model_id
            console.print(f"[cyan]⬇ Baixando '{target}' para o Cold Vault ({cfg.cold_hdd_dir})...[/cyan]")

            def show_prog(done: int, total: int) -> None:
                if total > 0:
                    pct = (done / total) * 100
                    sys.stdout.write(f"\r  Progresso: {done / (1024**3):.2f} / {total / (1024**3):.2f} GB ({pct:.1f}%)")
                    sys.stdout.flush()

            dest = runtime.vault.download_model(target, progress_cb=show_prog, activate_hot=True)
            console.print(f"\n[bold green]✓ Modelo pronto em: {dest}[/bold green]")
        return 0
    if cmd == "engine":
        if args.action == "install":
            console.print(f"[cyan]⚙️ Instalando binários llama-server (backend={args.backend})...[/cyan]")

            def _p(fname: str, done: int, total: int) -> None:
                if total > 0:
                    sys.stdout.write(f"\r  ⬇ {fname}: {done / (1024**2):.1f}/{total / (1024**2):.1f} MB")
                    sys.stdout.flush()

            exe = install_llama_engine(cfg, backend=args.backend, progress_cb=_p)
            console.print(f"\n[bold green]✓ Instalado com sucesso em: {exe}[/bold green]")
        elif args.action == "status":
            st = runtime.status()
            console.print(st)
        elif args.action == "cmd":
            render_hardware_and_split(cfg)
        elif args.action == "start":
            if runtime.find_llama_server_binary() is None:
                console.print("[cyan]Instalando llama-server automaticamente...[/cyan]")
                install_llama_engine(cfg, backend="auto")
            st = runtime.start()
            console.print(f"[green]✓ {st.message} ({st.endpoint})[/green]")
        elif args.action == "stop":
            runtime.stop()
            console.print("[yellow]Motor encerrado.[/yellow]")
        return 0

    interactive_chat(cfg, workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
