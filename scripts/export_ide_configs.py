"""Export ready-to-import configuration snippets for VS Code (Roo Code, Cline, Continue) & OpenCode."""

from __future__ import annotations

import json
from pathlib import Path

from nglai.config import NGLAIConfig


def export_configs(output_dir: Path | None = None) -> Path:
    cfg = NGLAIConfig.load()
    out = output_dir or (Path(cfg.base_dir) / ".nglai" / "ide_configs")
    out.mkdir(parents=True, exist_ok=True)
    gateway = f"http://{cfg.web_host}:{cfg.web_port}/v1"

    roo_cline = {
        "apiProvider": "openai",
        "openAiBaseUrl": gateway,
        "openAiApiKey": "nglai-local",
        "openAiModelId": cfg.active_model_id,
        "openAiCustomHeaders": {"X-NGLAI-RepoMap": "true"},
        "openAiStreamingEnabled": True,
    }
    (out / "roo_code_cline.json").write_text(json.dumps(roo_cline, indent=2), encoding="utf-8")

    continue_cfg = {
        "models": [
            {
                "title": f"New-Gen-Local-AI ({cfg.active_model_id})",
                "provider": "openai",
                "model": cfg.active_model_id,
                "apiBase": gateway,
                "apiKey": "nglai-local",
                "requestOptions": {"headers": {"X-NGLAI-RepoMap": "true"}},
            }
        ]
    }
    (out / "continue_config.json").write_text(json.dumps(continue_cfg, indent=2), encoding="utf-8")

    opencode_cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "nglai": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "New-Gen-Local-AI Triple-Stack",
                "options": {
                    "baseURL": gateway,
                    "apiKey": "nglai-local",
                    "headers": {"X-NGLAI-RepoMap": "true"},
                },
                "models": {
                    cfg.active_model_id: {
                        "name": cfg.active_model_id,
                    }
                },
            }
        },
    }
    (out / "opencode.json").write_text(json.dumps(opencode_cfg, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    dest = export_configs()
    print(f"Arquivos de integração IDE exportados em: {dest}")
