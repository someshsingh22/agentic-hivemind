#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-/mnt/localssd/namit/experigen-evals/.venv}"
source "$VENV/bin/activate"
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
source "$NVM_DIR/nvm.sh"
nvm use 24
hash -r
pi --version
exec "$VENV/bin/python" -u -m sonnet_dedup.hybrid \
  runs/calibration_paper_definitions_100/cmv/decomposed.json \
  runs/calibration_paper_definitions_100/cmv/plan.json \
  --config config.yaml --out runs/hybrid_paper_definitions/cmv --threshold 0.30
