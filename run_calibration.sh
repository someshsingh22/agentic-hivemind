#!/usr/bin/env bash
# Launch decomposition -> deduplication with the existing evaluation venv; follow its log.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

worker=0
if [[ "${1:-}" == --worker ]]; then
    worker=1
    shift
fi
tuning=0
if [[ "${1:-}" == --tune ]]; then
    tuning=1
    shift
fi
if (( $# > 2 )); then
    echo "Usage: bash run_calibration.sh [--tune] [envs/cmv.yaml] [config.yaml]" >&2
    exit 2
fi
env_file="${1:-envs/cmv.yaml}"
config_file="${2:-config.yaml}"
export VENV="${VENV:-/mnt/localssd/namit/experigen-evals/.venv}"
if [[ ! -f "$VENV/bin/activate" || ! -x "$VENV/bin/python" ]]; then
    echo "Evaluation venv not found: $VENV. Set VENV to the existing environment directory." >&2
    exit 2
fi
if [[ ! -r "$env_file" || ! -r "$config_file" ]]; then
    echo "Cannot read environment/config: $env_file | $config_file" >&2
    exit 2
fi
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

if (( worker )); then
    trap 'code=$?; printf "[%s] Calibration exited with status %s\n" "$(date -u +%FT%TZ)" "$code"' EXIT
    printf '[%s] Python: %s\n' "$(date -u +%FT%TZ)" "$VENV/bin/python"
    if (( tuning )); then
        printf '[%s] Starting shared decomposition and three-threshold tuning\n' "$(date -u +%FT%TZ)"
        "$VENV/bin/python" -u -m ssh tune "$env_file" --config "$config_file"
    else
        printf '[%s] Starting decomposition: %s | %s\n' "$(date -u +%FT%TZ)" "$env_file" "$config_file"
        "$VENV/bin/python" -u -m ssh decompose "$env_file" --config "$config_file"
        printf '[%s] Starting semantic-feature deduplication\n' "$(date -u +%FT%TZ)"
        "$VENV/bin/python" -u -m ssh dedup "$env_file" --config "$config_file"
    fi
    printf '[%s] Calibration complete. Stop here and review the artifacts; no annotation or SSH tests ran.\n' "$(date -u +%FT%TZ)"
    exit 0
fi

log_dir="${LOG_DIR:-$ROOT/runs/launchers}"
mkdir -p "$log_dir"
log="$log_dir/calibration_$(date -u +%Y%m%dT%H%M%S)_$$.log"
mode=()
if (( tuning )); then mode=(--tune); fi
nohup bash "$ROOT/run_calibration.sh" --worker "${mode[@]}" "$env_file" "$config_file" > "$log" 2>&1 < /dev/null &
pid=$!
printf 'Started calibration PID=%s\nLog: %s\nFollow again: tail -n 100 -F "%s"\n' "$pid" "$log" "$log"
printf 'Ctrl-C stops following the log; the nohup worker continues. Do not launch it again while it is running.\n'
trap 'printf "\nStopped following. Worker PID=%s; log=%s\n" "$pid" "$log"; exit 0' INT TERM
tail --pid="$pid" -n +1 -F "$log"
wait "$pid"
