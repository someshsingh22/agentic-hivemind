#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="${VENV:-/mnt/localssd/namit/experigen-evals/.venv}"
source "$VENV/bin/activate"
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
source "$NVM_DIR/nvm.sh"
nvm use 24
hash -r
node --version
pi --version

printf 'Starting decomposition only; no embeddings, merging, annotation or statistical tests.\n'
printf 'The agent is silent; each completed hypothesis appears in the configured output/decomposed/ directory.\n'
exec "$VENV/bin/python" -u -m ssh decompose envs/cmv.yaml --config config.yaml
