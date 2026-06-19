#!/usr/bin/env bash
# Удалить папки xframes, graph и streams на удалённом сервере.
#   DEPLOY_REMOTE      — хост (по умолчанию root@204.13.237.94)
#   REMOTE_XFRAMES_DIR — путь на сервере (по умолчанию /home/poly/xframes)
#   REMOTE_GRAPH_DIR   — путь graph на сервере (по умолчанию sibling к REMOTE_XFRAMES_DIR)
#   REMOTE_STREAMS_DIR — путь streams на сервере (по умолчанию sibling к REMOTE_XFRAMES_DIR)
#
# Запускайте откуда угодно, например:
#   bash deploy/clear-xframes.sh
#   bash /полный/путь/clear-xframes.sh

set -euo pipefail

REMOTE="${DEPLOY_REMOTE:-root@204.13.237.94}"
REMOTE_XFRAMES_DIR="${REMOTE_XFRAMES_DIR:-/home/poly/xframes}"

sibling_dir() {
  local path="${1%/}"
  local name="$2"
  local parent="${path%/*}"
  if [[ "$parent" == "$path" ]]; then
    printf '%s\n' "$name"
  elif [[ -z "$parent" ]]; then
    printf '/%s\n' "$name"
  else
    printf '%s/%s\n' "$parent" "$name"
  fi
}

REMOTE_GRAPH_DIR="${REMOTE_GRAPH_DIR:-$(sibling_dir "$REMOTE_XFRAMES_DIR" graph)}"
REMOTE_STREAMS_DIR="${REMOTE_STREAMS_DIR:-$(sibling_dir "$REMOTE_XFRAMES_DIR" streams)}"

echo "[clear-xframes] Удаляем $REMOTE:$REMOTE_XFRAMES_DIR, $REMOTE:$REMOTE_GRAPH_DIR и $REMOTE:$REMOTE_STREAMS_DIR …"
ssh "$REMOTE" "rm -rf '$REMOTE_XFRAMES_DIR' '$REMOTE_GRAPH_DIR' '$REMOTE_STREAMS_DIR'"
echo "[clear-xframes] Готово."
