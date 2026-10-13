#!/bin/bash
# Copy a folder from the pod to here through the RunPod SSH proxy (no scp there): tar.gz -> base64 -> rp.sh -> decode.
#   pod/fetch_results.sh [remote_dir] [local_dir]      default: /root/bfcl-results -> results/pod
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE="${1:-/root/bfcl-results}"; LOCAL="${2:-results/pod}"
mkdir -p "$LOCAL"
tmp=$(mktemp "${LOCAL%/}/.fetch.XXXXXX")
# build the archive once on the pod, then send it and its checksum (two separate tar runs could differ)
pod/rp.sh "cd '$(dirname "$REMOTE")' && tar czf /tmp/fetch.tgz --exclude=.file_locks '$(basename "$REMOTE")' && base64 -w0 /tmp/fetch.tgz; echo; md5sum /tmp/fetch.tgz; rm -f /tmp/fetch.tgz" > "$tmp"
b64=$(head -n 1 "$tmp"); want=$(sed -n 2p "$tmp" | cut -d' ' -f1)
echo "$b64" | base64 -d > "$tmp.tgz"
got=$(md5 -q "$tmp.tgz" 2>/dev/null || md5sum "$tmp.tgz" | cut -d' ' -f1)
if [ "$got" != "$want" ]; then echo "checksum mismatch ($got vs $want), nothing extracted" >&2; rm -f "$tmp" "$tmp.tgz"; exit 1; fi
tar xzf "$tmp.tgz" -C "$LOCAL"
rm -f "$tmp" "$tmp.tgz"
echo "copied $REMOTE -> $LOCAL/$(basename "$REMOTE") ($(du -sh "$LOCAL/$(basename "$REMOTE")" | cut -f1), checksum ok)"
