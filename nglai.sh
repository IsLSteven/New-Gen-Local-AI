#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NGLAI_HOME="$ROOT_DIR"
export HF_HOME="$ROOT_DIR/.cache/hf"
export PIP_CACHE_DIR="$ROOT_DIR/.cache/pip"

if [ ! -x "$ROOT_DIR/.venv/bin/python" ]; then
  echo "[New-Gen-Local-AI] Criando ambiente virtual isolado..."
  mkdir -p "$ROOT_DIR/.cache/tmp"
  python3 -m venv "$ROOT_DIR/.venv"
  "$ROOT_DIR/.venv/bin/pip" install -e "$ROOT_DIR"
fi

exec "$ROOT_DIR/.venv/bin/python" -m nglai.cli "$@"
