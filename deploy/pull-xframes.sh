#!/usr/bin/env bash
# Скачать `$REMOTE_XFRAMES_DIR` с удалённого сервера в `$LOCAL_XFRAMES_DIR`.
#   FETCH_REMOTE       — хост (по умолчанию root@204.13.237.94)
#   REMOTE_XFRAMES_DIR — путь на сервере (по умолчанию /home/poly/xframes)
#   LOCAL_XFRAMES_DIR  — путь локально (по умолчанию XFRAMES_DIR из .env,
#                        иначе <repo>/xframes)
#   DELETE=1           — удалять локальные файлы, которых уже нет на сервере
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

mkdir -p "$LOCAL_XFRAMES_DIR"

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
echo "[fetch-xframes] Готово."
