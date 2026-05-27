#!/usr/bin/env bash
# Pack Wan2.2 repository source + local tools; exclude dataset (release/), checkpoints (action_head_runs/), venv, caches, and large generated media.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_OUT="$(dirname "${REPO_ROOT}")/Wan2.2_algorithm_$(date +%Y%m%d_%H%M%S).tar.gz"
OUT="${1:-$DEFAULT_OUT}"

cd "$(dirname "${REPO_ROOT}")"
ARCHIVE_NAME="$(basename "${OUT}")"
ARCHIVE_DIR="$(dirname "${OUT}")"
mkdir -p "${ARCHIVE_DIR}"

REPO_BASENAME="$(basename "${REPO_ROOT}")"

echo "Packing ${REPO_ROOT} -> ${OUT}"
echo "Excludes: release/ (dataset), action_head_runs/ (checkpoints), .venv, Wan2.2-* model dirs, caches, .git"

# GNU tar: excludes are path prefixes as stored in archive (REPO_BASENAME/...)
tar -czf "${OUT}" \
  --exclude="${REPO_BASENAME}/release" \
  --exclude="${REPO_BASENAME}/action_head_runs" \
  --exclude="${REPO_BASENAME}/.venv" \
  --exclude="${REPO_BASENAME}/Wan2.2-TI2V-5B" \
  --exclude="${REPO_BASENAME}/Wan2.2-T2V-A14B" \
  --exclude="${REPO_BASENAME}/Wan2.2-I2V-A14B" \
  --exclude="${REPO_BASENAME}/Wan2.2-S2V-14B" \
  --exclude="${REPO_BASENAME}/Wan2.2-Animate-14B" \
  --exclude="${REPO_BASENAME}/__pycache__" \
  --exclude="${REPO_BASENAME}/*/__pycache__" \
  --exclude="${REPO_BASENAME}/*/*/__pycache__" \
  --exclude="${REPO_BASENAME}/*/*/*/__pycache__" \
  --exclude="${REPO_BASENAME}/.pytest_cache" \
  --exclude="${REPO_BASENAME}/.mypy_cache" \
  --exclude="${REPO_BASENAME}/.ruff_cache" \
  --exclude="${REPO_BASENAME}/.git" \
  --exclude="${REPO_BASENAME}/*.mp4" \
  --exclude="${REPO_BASENAME}/*.pth" \
  --exclude="${REPO_BASENAME}/*.pt" \
  --exclude="${REPO_BASENAME}/*.safetensors" \
  --exclude="${REPO_BASENAME}/*.ckpt" \
  "${REPO_BASENAME}"

ls -lh "${OUT}"
echo "Done."
