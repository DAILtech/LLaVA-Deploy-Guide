#!/bin/bash
# Script to download LLaVA model weights from Hugging Face.
# Usage: bash download_model.sh <model_version> [--hf-mirror]
# Example: bash download_model.sh llava-1.5-7b --hf-mirror

set -e

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_version> [--hf-mirror]"
  echo "Example: $0 llava-1.5-7b --hf-mirror"
  exit 1
fi

MODEL_VERSION=$1
USE_HF_MIRROR=false
if [[ "$2" == "--hf-mirror" ]]; then
  USE_HF_MIRROR=true
fi

# Map model_version to Hugging Face repository
HF_REPO=""
case "${MODEL_VERSION}" in
  "llava-1.5-7b") HF_REPO="liuhaotian/llava-v1.5-7b" ;;
  "llava-1.5-13b") HF_REPO="liuhaotian/llava-v1.5-13b" ;;
  "llava-1.6-7b") HF_REPO="liuhaotian/llava-v1.6-vicuna-7b" ;;
  "llava-1.6-13b") HF_REPO="liuhaotian/llava-v1.6-vicuna-13b" ;;
  *)
    echo "[Error] Unknown model version: ${MODEL_VERSION}"
    exit 1
    ;;
esac

# Determine download URL (git clone via HTTPS)
BASE_URL="https://huggingface.co"
if $USE_HF_MIRROR; then
  BASE_URL="https://hf-mirror.com"
  echo "[Info] Using Hugging Face mirror: ${BASE_URL}"
fi
REPO_URL="${BASE_URL}/${HF_REPO}"

# Create models directory if not exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(realpath "${SCRIPT_DIR}/..")"
mkdir -p "${ROOT_DIR}/models"

TARGET_DIR="${ROOT_DIR}/models/${MODEL_VERSION}"

echo "[Info] Downloading model '${MODEL_VERSION}' from ${REPO_URL} ..."
if [[ -d "${TARGET_DIR}" ]]; then
  echo "[Warning] Target directory ${TARGET_DIR} already exists. The model may have been downloaded. Skipping."
  exit 0
fi

git lfs install 2>/dev/null || true   # ensure git-lfs is initialized
git clone "${REPO_URL}" "${TARGET_DIR}" || {
    echo "[Error] git clone failed. If this is a permission issue, ensure you've accepted the model license and have a valid Hugging Face token."
    exit 1
}
echo "[Info] Model files for '${MODEL_VERSION}' downloaded to ${TARGET_DIR}"
