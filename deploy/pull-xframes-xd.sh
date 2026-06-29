#!/usr/bin/env bash
# Скачать только свежие `$REMOTE_XFRAMES_DIR` и sibling `graph`/`streams`
# с удалённого сервера.
#   DAYS                — сколько последних суток тянуть по mtime на remote
#                         (по умолчанию 2: последние 48 часов)
#   FETCH_REMOTE        — хост (по умолчанию root@204.13.237.94)
#   REMOTE_XFRAMES_DIR  — путь на сервере (по умолчанию /home/poly/xframes)
#   REMOTE_GRAPH_DIR    — путь graph на сервере (по умолчанию sibling к REMOTE_XFRAMES_DIR)
#   REMOTE_STREAMS_DIR  — путь streams на сервере (по умолчанию sibling к REMOTE_XFRAMES_DIR)
#   LOCAL_XFRAMES_DIR   — путь локально (по умолчанию XFRAMES_DIR из .env,
#                         иначе <repo>/xframes)
#   LOCAL_GRAPH_DIR     — путь graph локально (по умолчанию sibling к LOCAL_XFRAMES_DIR)
#   LOCAL_STREAMS_DIR   — путь streams локально (по умолчанию sibling к LOCAL_XFRAMES_DIR)
#
# Запускайте откуда угодно, например:
#   bash deploy/pull-xframes-xd.sh
#   DAYS=3 bash deploy/pull-xframes-xd.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE="${FETCH_REMOTE:-root@204.13.237.94}"
REMOTE_XFRAMES_DIR="${REMOTE_XFRAMES_DIR:-/home/poly/xframes}"
DAYS="${DAYS:-2}"

if ! [[ "$DAYS" =~ ^[0-9]+$ ]] || [[ "$DAYS" -lt 1 ]]; then
  echo "[fetch-xframes-xd] DAYS must be a positive integer, got: $DAYS" >&2
  exit 2
fi

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
RSYNC_OPTS=(-avzhP)
MINUTES=$((DAYS * 24 * 60))

if [[ "${DELETE:-0}" == "1" ]]; then
  echo "[fetch-xframes-xd] DELETE=1 игнорируется: скрипт синхронизирует только file-list за последние ${DAYS} суток" >&2
fi

TMP_FILES=()
cleanup() {
  local path
  for path in "${TMP_FILES[@]}"; do
    rm -f "$path"
  done
}
trap cleanup EXIT

fetch_recent_tree() {
  local label="$1"
  local remote_dir="$2"
  local local_dir="$3"
  local list_file
  local count

  list_file="$(mktemp)"
  TMP_FILES+=("$list_file")

  echo "[fetch-${label}-xd] build remote file-list: ${REMOTE}:${remote_dir}/ newer than ${MINUTES}m"
  ssh "$REMOTE" "cd \"$remote_dir\" && find . -type f -mmin -$MINUTES -print | LC_ALL=C sort" > "$list_file"

  count="$(wc -l < "$list_file" | tr -d ' ')"
  echo "[fetch-${label}-xd] files: $count"
  if [[ "$count" == "0" ]]; then
    return
  fi

  echo "[fetch-${label}-xd] ${REMOTE}:${remote_dir}/  →  $local_dir/"
  rsync "${RSYNC_OPTS[@]}" --files-from="$list_file" "${REMOTE}:${remote_dir}/" "$local_dir/"
}

echo "[fetch-xframes-xd] window: last $DAYS day(s) = ${MINUTES} minutes by remote mtime"

fetch_recent_tree xframes "$REMOTE_XFRAMES_DIR" "$LOCAL_XFRAMES_DIR"
fetch_recent_tree graph "$REMOTE_GRAPH_DIR" "$LOCAL_GRAPH_DIR"
fetch_recent_tree streams "$REMOTE_STREAMS_DIR" "$LOCAL_STREAMS_DIR"
echo "[fetch-xframes-xd] Готово."
