#!/usr/bin/env bash
# Локальная сборка, rsync в $DEPLOY_DIR, остановка старого poly, старт через nohup.
#   DEPLOY_REMOTE — хост (по умолчанию root@204.13.237.94)
#   DEPLOY_DIR    — каталог на сервере (по умолчанию /home/poly)
#
# Запускайте откуда угодно, например:
#   bash deploy/deploy.sh
#   bash /полный/путь/deploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
cargo build --profile release
cp "$REPO_ROOT/target/release/poly" "$SCRIPT_DIR/poly"
cd "$SCRIPT_DIR"

REMOTE="${DEPLOY_REMOTE:-root@204.13.237.94}"
REMOTE_DIR="${DEPLOY_DIR:-/home/poly}"

rsync -avz ./ "$REMOTE:$REMOTE_DIR/"

# shellcheck disable=SC2029
echo "[deploy] Останавливаем существующий poly на $REMOTE (если был)…"
ssh "$REMOTE" "bash -s" -- "$REMOTE_DIR" <<'EOF'
# fuser/pkill часто возвращают неноль, если нечего убивать.
D="${1:-}"
B="${D}/poly"
if [[ -z "$D" || ! -d "$D" ]]; then
  echo "[deploy/remote] пропуск stop: нет каталога $D" >&2
  exit 0
fi
if [[ -f "$B" ]]; then
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -TERM "$B" 2>/dev/null || true
    sleep 0.4
    fuser -k -9 "$B" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -t "$B" 2>/dev/null || true)
    if [[ -n "${pids:-}" ]]; then
      # shellcheck disable=SC2086
      kill -TERM $pids 2>/dev/null || true
      sleep 0.4
      pids2=$(lsof -t "$B" 2>/dev/null || true)
      if [[ -n "${pids2:-}" ]]; then
        # shellcheck disable=SC2086
        kill -9 $pids2 2>/dev/null || true
      fi
    fi
  else
    (cd "$D" && pkill -TERM -f "^\./poly" 2>/dev/null) || true
    sleep 0.4
    (cd "$D" && pkill -9 -f "^\./poly" 2>/dev/null) || true
  fi
fi
sleep 0.2
EOF

echo "[deploy] Старт на $REMOTE (nohup)…"
ssh "$REMOTE" "cd $REMOTE_DIR && chmod +x start.sh poly 2>/dev/null || true && (nohup ./start.sh >> nohup-poly.log 2>&1 < /dev/null &)"
echo "[deploy] Готово. Лог: $REMOTE_DIR/nohup-poly.log  →  ssh $REMOTE 'tail -f $REMOTE_DIR/nohup-poly.log'"
