#!/usr/bin/env bash
# Provision the local notebook/control-plane environment on Azure ML compute.
#
# The venv is created under /mnt (ephemeral scratch on the compute VM) because
# ~/cloudfiles is an Azure Files share that makes import-heavy Python startup
# significantly slower. /mnt is wiped when the compute stops/restarts, so this
# script is idempotent — re-run it after every compute (re)start.
#
# Usage:
#   bash scripts/setup_env.sh              # default venv path + kernel name
#   VENV_DIR=/mnt/tmp/venvs/my-env bash scripts/setup_env.sh
#   KERNEL_NAME=raft-py312 KERNEL_DISPLAY_NAME="Python 3.12.10 (raft /mnt)" \
#     bash scripts/setup_env.sh

set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12.10}"
VENV_DIR="${VENV_DIR:-/mnt/tmp/venvs/raft-py312-mnt}"
KERNEL_NAME="${KERNEL_NAME:-raft-py312-mnt}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python ${PYTHON_VERSION} (raft /mnt)}"

# uv installs Pythons and caches wheels; pin both to /mnt so nothing lands on
# the slow Azure Files share.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/mnt/tmp/uv-python}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/tmp/uv-cache}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[setup_env] repo:           ${REPO_ROOT}"
echo "[setup_env] python version: ${PYTHON_VERSION}"
echo "[setup_env] venv dir:       ${VENV_DIR}"
echo "[setup_env] uv python dir:  ${UV_PYTHON_INSTALL_DIR}"
echo "[setup_env] uv cache dir:   ${UV_CACHE_DIR}"

mkdir -p "$(dirname "${VENV_DIR}")" "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup_env] installing uv into the current interpreter"
    python -m pip install --quiet --upgrade uv
fi

echo "[setup_env] ensuring CPython ${PYTHON_VERSION} is available via uv"
uv python install "${PYTHON_VERSION}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[setup_env] creating venv at ${VENV_DIR}"
    uv venv --python "${PYTHON_VERSION}" --seed "${VENV_DIR}"
else
    echo "[setup_env] reusing existing venv at ${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"

echo "[setup_env] installing requirements-azureml.txt"
uv pip install --python "${VENV_PY}" -r "${REPO_ROOT}/requirements-azureml.txt"

if [[ -f "${REPO_ROOT}/requirements.txt" ]]; then
    echo "[setup_env] installing requirements.txt"
    uv pip install --python "${VENV_PY}" -r "${REPO_ROOT}/requirements.txt"
fi

echo "[setup_env] installing ipykernel and registering Jupyter kernelspec"
uv pip install --python "${VENV_PY}" ipykernel
"${VENV_PY}" -m ipykernel install \
    --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY_NAME}"

echo
echo "[setup_env] done."
echo "  Activate:      source ${VENV_DIR}/bin/activate"
echo "  Notebook kernel: ${KERNEL_DISPLAY_NAME}  (id: ${KERNEL_NAME})"
echo "  Persist uv paths in ~/.bashrc:"
echo "    export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR}"
echo "    export UV_CACHE_DIR=${UV_CACHE_DIR}"
