#!/usr/bin/env bash
# Скачать `$REMOTE_XFRAMES_DIR` и sibling `graph`/`streams` с удалённого сервера.
#   FETCH_REMOTE        — хост (по умолчанию root@204.13.237.94)
#   REMOTE_XFRAMES_DIR  — путь на сервере (по умолчанию /home/poly/xframes)
#   REMOTE_GRAPH_DIR    — путь graph на сервере (по умолчанию sibling к REMOTE_XFRAMES_DIR)
#   REMOTE_STREAMS_DIR  — путь streams на сервере (по умолчанию sibling к REMOTE_XFRAMES_DIR)
#   LOCAL_XFRAMES_DIR   — путь локально (по умолчанию XFRAMES_DIR из .env,
#                         иначе <repo>/xframes)
#   LOCAL_GRAPH_DIR     — путь graph локально (по умолчанию sibling к LOCAL_XFRAMES_DIR)
#   LOCAL_STREAMS_DIR   — путь streams локально (по умолчанию sibling к LOCAL_XFRAMES_DIR)
#   DELETE=1            — удалять локальные файлы, которых уже нет на сервере
#                        (по умолчанию выключено)
#
# Запускайте откуда угодно, например:
#   bash deploy/pull-xframes.sh
#   bash /полный/путь/pull-xframes.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE="${FETCH_REMOTE:-root@204.13.237.94}"
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

ENV_XFRAMES_DIR=""
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS= read -r line; do
    line="${line%$'\r'}"
    [[ "$line" == XFRAMES_DIR=* ]] || continue
    ENV_XFRAMES_DIR="${line#XFRAMES_DIR=}"
    ENV_XFRAMES_DIR="${ENV_XFRAMES_DIR%\"}"
    ENV_XFRAMES_DIR="${ENV_XFRAMES_DIR#\"}"
    break
  done < "$REPO_ROOT/.env"
fi
LOCAL_XFRAMES_DIR="${LOCAL_XFRAMES_DIR:-${ENV_XFRAMES_DIR:-$REPO_ROOT/xframes}}"
REMOTE_GRAPH_DIR="${REMOTE_GRAPH_DIR:-$(sibling_dir "$REMOTE_XFRAMES_DIR" graph)}"
REMOTE_STREAMS_DIR="${REMOTE_STREAMS_DIR:-$(sibling_dir "$REMOTE_XFRAMES_DIR" streams)}"
LOCAL_GRAPH_DIR="${LOCAL_GRAPH_DIR:-$(sibling_dir "$LOCAL_XFRAMES_DIR" graph)}"
LOCAL_STREAMS_DIR="${LOCAL_STREAMS_DIR:-$(sibling_dir "$LOCAL_XFRAMES_DIR" streams)}"

mkdir -p "$LOCAL_XFRAMES_DIR" "$LOCAL_GRAPH_DIR" "$LOCAL_STREAMS_DIR"

# rsync flags:
#   -a  — архив (рекурсивно, права/симлинки/времена)
#   -v  — вывод
#   -z  — компрессия по сети
#   -h  — human-readable размеры
#   -P  — прогресс + докачка частично переданных файлов
# Завершающий слэш у источника `…/xframes/` (после quoting): копируем содержимое внутрь
# целевой папки, без вложения второго уровня (`xframes/xframes/...`).
RSYNC_OPTS=(-avzhP)
if [[ "${DELETE:-0}" == "1" ]]; then
  RSYNC_OPTS+=(--delete)
  echo "[fetch-xframes] DELETE=1 → локальные файлы вне сервера будут удалены"
fi

echo "[fetch-xframes] $REMOTE:$REMOTE_XFRAMES_DIR/  →  $LOCAL_XFRAMES_DIR/"
rsync "${RSYNC_OPTS[@]}" "$REMOTE:$REMOTE_XFRAMES_DIR/" "$LOCAL_XFRAMES_DIR/"

echo "[fetch-graph] $REMOTE:$REMOTE_GRAPH_DIR/  →  $LOCAL_GRAPH_DIR/"
rsync "${RSYNC_OPTS[@]}" "$REMOTE:$REMOTE_GRAPH_DIR/" "$LOCAL_GRAPH_DIR/"

echo "[fetch-streams] $REMOTE:$REMOTE_STREAMS_DIR/  →  $LOCAL_STREAMS_DIR/"
rsync "${RSYNC_OPTS[@]}" "$REMOTE:$REMOTE_STREAMS_DIR/" "$LOCAL_STREAMS_DIR/"
echo "[fetch-xframes] Готово."
