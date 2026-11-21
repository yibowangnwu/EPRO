#!/usr/bin/env bash
set -euo pipefail
EPRO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VERL_ROOT=${VERL_ROOT:-${EPRO_ROOT}/third_party/verl}
VERL_COMMIT=4b26c788d41ce932ce7706c5ee8e249ab52f4c51
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
if [[ ! -e "${VERL_ROOT}" ]]; then
    mkdir -p -- "$(dirname -- "${VERL_ROOT}")"
    git clone https://github.com/verl-project/verl.git "${VERL_ROOT}"
    git -C "${VERL_ROOT}" checkout --detach "${VERL_COMMIT}"
fi
uv run --no-project --python 3.12 python "${EPRO_ROOT}/scripts/install_verl.py" "${VERL_ROOT}"
cd -- "${VERL_ROOT}"
# Use verl's committed lockfile, including its CUDA 13 wheel indexes.
uv sync --frozen --python 3.12 --all-packages --extra fsdp --extra vllm
printf 'Environment ready. Use %s/.venv/bin/python to launch training.\n' "${VERL_ROOT}"
