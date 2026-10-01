#!/usr/bin/env bash
set -euo pipefail

# Select teacher/student models, dataset and evaluation stage in configs/analyze.yaml.
# Explicit student/checker GPU lists override this value. Each GPU holds one replica.
export CUDA_VISIBLE_DEVICES=2,3
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

# Set evaluation.offline in YAML or pass evaluation.offline=true on the CLI.
# Python applies it to dataset loading and the model workers' HF offline variables.

cd "${PROJECT_DIR}"
# Launch one evaluator; student and checker stages use the configured GPU replicas.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_COMMAND=("${PYTHON_BIN}")
else
  PYTHON_COMMAND=(uv run python)
fi
exec "${PYTHON_COMMAND[@]}" "${SCRIPT_DIR}/analyze_simulation_result.py" "$@"
