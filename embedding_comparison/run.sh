#!/usr/bin/env bash
# Compare frozen semantic feature representations and follow the detached worker log.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
worker=0
if [[ "${1:-}" == --worker ]]; then worker=1; shift; fi
if (( $# > 3 )); then
    echo "Usage: bash embedding_comparison/run.sh [DECOMPOSED_JSON] [CONFIG_YAML] [OUTPUT_DIR]" >&2
    exit 2
fi
decomposed="${1:-runs/calibration_v8_examples/cmv/decomposed.json}"
config="${2:-config.yaml}"
out="${3:-runs/embedding_inputs/cmv}"
export VENV="${VENV:-/mnt/localssd/namit/experigen-evals/.venv}"
if [[ ! -f "$VENV/bin/activate" || ! -x "$VENV/bin/python" ]]; then
    echo "Evaluation venv not found: $VENV. Set VENV to the existing environment directory." >&2
    exit 2
fi
if [[ ! -r "$decomposed" || ! -r "$config" ]]; then
    echo "Cannot read input/config: $decomposed | $config" >&2
    exit 2
fi
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
if (( worker )); then
    trap 'code=$?; printf "[%s] Embedding comparison exited with status %s\n" "$(date -u +%FT%TZ)" "$code"' EXIT
    printf '[%s] Comparing frozen feature representations: %s\n' "$(date -u +%FT%TZ)" "$decomposed"
    "$VENV/bin/python" -u -m embedding_comparison run "$decomposed" --config "$config" --out "$out"
    printf '[%s] Complete. No decomposition, row annotation or SSH statistics ran.\n' "$(date -u +%FT%TZ)"
    exit 0
fi
log_dir="${LOG_DIR:-$ROOT/runs/launchers}"
mkdir -p "$log_dir"
log="$log_dir/embedding_inputs_$(date -u +%Y%m%dT%H%M%S)_$$.log"
nohup bash "$ROOT/embedding_comparison/run.sh" --worker "$decomposed" "$config" "$out" > "$log" 2>&1 < /dev/null &
pid=$!
printf 'Started embedding comparison PID=%s\nLog: %s\nFollow again: tail -n 100 -F "%s"\n' "$pid" "$log" "$log"
printf 'Ctrl-C stops following the log; the nohup worker continues. Do not launch it again while it is running.\n'
trap 'printf "\nStopped following. Worker PID=%s; log=%s\n" "$pid" "$log"; exit 0' INT TERM
tail --pid="$pid" -n +1 -F "$log"
wait "$pid"
