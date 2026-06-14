#!/usr/bin/env bash
# Удалить папки xframes и graph на удалённом сервере.
#   DEPLOY_REMOTE      — хост (по умолчанию root@204.13.237.94)
#   REMOTE_XFRAMES_DIR — путь на сервере (по умолчанию /home/poly/xframes)
#   REMOTE_GRAPH_DIR   — путь на сервере (по умолчанию /home/poly/graph)
#
# Запускайте откуда угодно, например:
#   bash deploy/clear-xframes.sh
#   bash /полный/путь/clear-xframes.sh

set -euo pipefail

REMOTE="${DEPLOY_REMOTE:-root@204.13.237.94}"
REMOTE_XFRAMES_DIR="${REMOTE_XFRAMES_DIR:-/home/poly/xframes}"
REMOTE_GRAPH_DIR="${REMOTE_GRAPH_DIR:-/home/poly/graph}"

echo "[clear-xframes] Удаляем $REMOTE:$REMOTE_XFRAMES_DIR и $REMOTE:$REMOTE_GRAPH_DIR …"
ssh "$REMOTE" "rm -rf '$REMOTE_XFRAMES_DIR' '$REMOTE_GRAPH_DIR'"
echo "[clear-xframes] Готово."
