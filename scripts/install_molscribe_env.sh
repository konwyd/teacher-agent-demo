#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${MOLSCRIBE_VENV:-${PROJECT_DIR}/.venv-molscribe}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_INDEX_URL_MOLSCRIBE="${MOLSCRIBE_PIP_INDEX_URL:-https://pypi.org/simple}"

if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}" 2>/dev/null; then
  # Some minimal Ubuntu images ship Python without python3-venv/ensurepip.
  # virtualenv is a user-space fallback and avoids requiring sudo here.
  "${PYTHON_BIN}" -m pip install --user "virtualenv>=20.26"
  "${PYTHON_BIN}" -m virtualenv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

# MolScribe 1.1.1 pins torch<2.0. Keep this legacy stack isolated from the
# Teacher Agent runtime, which currently uses PyTorch 2.x for local inference.
"${VENV_DIR}/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  "torch==1.13.1+cu117" "torchvision==0.14.1+cu117"
# Install the declared runtime dependencies explicitly, then install MolScribe
# without dependency resolution. Its package metadata depends on the obsolete
# distribution name `rdkit-pypi`; modern PyPI publishes maintained wheels as
# `rdkit`, and installing both distributions into one environment is unsafe.
"${VENV_DIR}/bin/python" -m pip install \
  --index-url "${PIP_INDEX_URL_MOLSCRIBE}" \
  "pandas" \
  "numpy==1.26.4" \
  "opencv-python==4.8.1.78" \
  "opencv-python-headless==4.8.1.78" \
  "SmilesPE==0.0.3" \
  "OpenNMT-py==2.2.0" \
  "albumentations==1.1.0" \
  "timm==0.4.12" \
  "rdkit==2025.9.6" \
  "huggingface_hub>=0.20,<1" \
  "fastprogress==1.0.3"
"${VENV_DIR}/bin/python" -m pip install --no-deps "MolScribe==1.1.1"

"${VENV_DIR}/bin/python" - <<'PY'
from molscribe import MolScribe  # noqa: F401
from rdkit import rdBase
import torch

print(f"MolScribe import: OK")
print(f"RDKit: {rdBase.rdkitVersion}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY

printf 'MolScribe environment ready: %s\n' "${VENV_DIR}"
