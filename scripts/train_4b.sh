#!/usr/bin/env bash
set -euo pipefail
EPRO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export VERL_ROOT=${VERL_ROOT:-${EPRO_ROOT}/third_party/verl}
EPRO_PYTHON=${EPRO_PYTHON:-${VERL_ROOT}/.venv/bin/python}
exec "${EPRO_PYTHON}" "${EPRO_ROOT}/scripts/train.py" "$@"
