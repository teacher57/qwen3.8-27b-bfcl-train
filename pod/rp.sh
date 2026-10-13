#!/bin/bash
# Run a command on the pod through RunPod's SSH proxy and print only its output.
# The proxy (ssh.runpod.io) gives an interactive shell only: no scp, no "ssh host cmd". This wraps the
# command in markers and cuts out the banner and prompt noise.
#   ./rp.sh 'nvidia-smi; tail /root/pipeline.log'
#   ./rp.sh --put local_file /root/remote_file      # upload a file (base64 through the shell)
# Set RP_TARGET to the pod's proxy address (from the RunPod "Connect" panel).
RP_TARGET="${RP_TARGET:-<pod-id>-<n>@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"

if [ "$1" = "--put" ]; then
  b64=$(base64 < "$2" | tr -d '\n')
  set -- "echo $b64 | base64 -d > '$3' && chmod +x '$3' && md5sum '$3'"
fi

marker="__RP_$$_$RANDOM"
cmd_b64=$(printf '%s' "$1" | base64 | tr -d '\n')
ssh -tt -o ConnectTimeout=20 -o LogLevel=ERROR -i "$KEY" "$RP_TARGET" 2>/dev/null <<EOF | LC_ALL=C tr -d '\r' | LC_ALL=C sed -n "/^${marker}_START\$/,/^${marker}_END/p" | sed '1d;$d'
stty -echo 2>/dev/null; export PS1=''
echo ${marker}_START; echo $cmd_b64 | base64 -d | bash 2>&1; echo ${marker}_END
exit
EOF
