#!/bin/bash
# venv on the container disk (/dev/shm is noexec), uv temp + cache on /dev/shm (the disk is too small for both)
export PATH=/root/.local/bin:$PATH UV_LINK_MODE=copy TMPDIR=/dev/shm/uvtmp UV_CACHE_DIR=/dev/shm/uvcache
uv venv /root/bfcl-venv -p 3.12 -q && uv pip install -p /root/bfcl-venv -q -e /root/gorilla/berkeley-function-call-leaderboard soundfile && echo BFCL_DONE || echo BFCL_FAILED
