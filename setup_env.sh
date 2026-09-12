#!/usr/bin/env bash
# Create a Python 3.9 virtual environment with uv (https://astral.sh/uv) and install See2Act in editable mode.
# Usage: bash setup_env.sh            (CUDA 12.1 wheels; set TORCH_INDEX for another CUDA version)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
[ -d "$ROOT/.venv" ] || uv venv --python 3.9 "$ROOT/.venv"
PY="$ROOT/.venv/bin/python"
uv pip install --python "$PY" --index-url "$TORCH_INDEX" torch==2.1.1 torchvision==0.16.1
uv pip install --python "$PY" -e "$ROOT"
echo "Done. Activate with: source $ROOT/.venv/bin/activate"
