#!/usr/bin/env bash
# Container entrypoint: make sure the UPN checkpoint is present, then run the command.
set -euo pipefail

UPN_CKPT="${UPN_CKPT:-/app/resources/upn_large.pth}"
UPN_CKPT_URL="${UPN_CKPT_URL:-https://github.com/IDEA-Research/ChatRex/releases/download/upn-large/upn_large.pth}"

if [ "${SKIP_UPN_DOWNLOAD:-0}" != "1" ] && [ ! -s "$UPN_CKPT" ]; then
    echo "[entrypoint] UPN checkpoint not found at $UPN_CKPT; downloading (~875 MB) ..."
    mkdir -p "$(dirname "$UPN_CKPT")"
    curl -fL --retry 5 --retry-delay 3 -o "$UPN_CKPT.part" "$UPN_CKPT_URL"
    mv "$UPN_CKPT.part" "$UPN_CKPT"
    echo "[entrypoint] saved $UPN_CKPT"
fi

if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[entrypoint] WARNING: no CUDA device visible. Run with '--gpus all' (docker run)" >&2
    echo "[entrypoint]          or keep the 'deploy.resources' block in docker-compose.yml." >&2
fi

exec "$@"
