#!/usr/bin/env bash
set -euo pipefail

# Select configs/experiment/<name>.yaml here; no CLI arguments are needed.
# Options: local_only / mixed / api_only
EXPERIMENT="local_only"

# Explicit role GPU lists override this environment value. Each GPU holds one replica.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

export OMP_NUM_THREADS=4

export VLLM_CONFIGURE_LOGGING=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "${SCRIPT_DIR}")"
export HF_HOME="${PROJECT_DIR}/cache_dir"
export TZ="${TZ:-Asia/Seoul}"

TOKEN_FILE="${PROJECT_DIR}/huggingface_access_token.txt"
if [[ -z "${HF_TOKEN:-}" && -f "${TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(< "${TOKEN_FILE}")"
fi

is_offline="false"

export HF_HUB_OFFLINE=${is_offline}
export TRANSFORMERS_OFFLINE=${is_offline}
export HF_DATASETS_OFFLINE=${is_offline}

cd "${PROJECT_DIR}"
# Launch one simulator; it distributes role batches across persistent GPU replicas.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_COMMAND=("${PYTHON_BIN}")
else
  PYTHON_COMMAND=(uv run python)
fi
EXPERIMENT_ARGS=("experiment=${EXPERIMENT}")
for argument in "$@"; do
  case "${argument}" in
    experiment=*|+experiment=*|++experiment=*|~experiment|~experiment=*)
      EXPERIMENT_ARGS=()
      break
      ;;
  esac
done
exec "${PYTHON_COMMAND[@]}" "${SCRIPT_DIR}/tutoring_simulator.py" \
  "${EXPERIMENT_ARGS[@]}" "$@"
