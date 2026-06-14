#!/usr/bin/env bash
# Локальная сборка (CPU-only, без CUDA), гарантированная остановка старого poly,
# rsync в $DEPLOY_DIR, старт текущего ./poly через nohup без start.sh.
#   DEPLOY_REMOTE — хост (по умолчанию root@204.13.237.94)
#   DEPLOY_DIR    — каталог на сервере (по умолчанию /home/poly)
#
# На сервере нет GPU — собираем с --no-default-features (локально по умолчанию cuda).
# Запускайте откуда угодно, например:
#   bash deploy/deploy.sh
#   bash /полный/путь/deploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE="${DEPLOY_REMOTE:-root@204.13.237.94}"
REMOTE_DIR="${DEPLOY_DIR:-/home/poly}"

remote_stop_poly() {
  # shellcheck disable=SC2029
  ssh "$REMOTE" "bash -s" -- "$REMOTE_DIR" <<'EOF'
set -euo pipefail
D="${1:-}"
D="${D%/}"
B="$D/poly"

if [[ -z "$D" ]]; then
  echo "[deploy/remote] stop failed: empty DEPLOY_DIR" >&2
  exit 1
fi
if [[ ! -d "$D" ]]; then
  echo "[deploy/remote] stop: каталога $D нет, пропуск"
  exit 0
fi

find_poly_pids() {
  local proc pid exe cwd cmd
  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue

    exe="$(readlink "$proc/exe" 2>/dev/null || true)"
    cwd="$(readlink "$proc/cwd" 2>/dev/null || true)"
    cmd="$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)"
    [[ -z "$cmd" ]] && continue

    if [[ "$exe" == "$B" || "$exe" == "$B (deleted)" ]]; then
      printf '%s\n' "$pid"
      continue
    fi
    if [[ "$cmd" == "$B" || "$cmd" == "$B "* ]]; then
      printf '%s\n' "$pid"
      continue
    fi
    if [[ "$cwd" == "$D" && ("$cmd" == "./poly" || "$cmd" == "./poly "*) ]]; then
      printf '%s\n' "$pid"
      continue
    fi
  done | sort -u
}

mapfile -t pids < <(find_poly_pids)
if (( ${#pids[@]} == 0 )); then
  echo "[deploy/remote] stop: poly не запущен"
  exit 0
fi

echo "[deploy/remote] TERM poly pid(s): ${pids[*]}"
kill -TERM "${pids[@]}" 2>/dev/null || true

for _ in {1..20}; do
  sleep 0.25
  mapfile -t pids < <(find_poly_pids)
  if (( ${#pids[@]} == 0 )); then
    echo "[deploy/remote] stop: poly остановлен"
    exit 0
  fi
done

echo "[deploy/remote] KILL poly pid(s): ${pids[*]}" >&2
kill -KILL "${pids[@]}" 2>/dev/null || true
sleep 0.2
mapfile -t pids < <(find_poly_pids)
if (( ${#pids[@]} != 0 )); then
  echo "[deploy/remote] stop failed: живы pid(s): ${pids[*]}" >&2
  exit 1
fi
echo "[deploy/remote] stop: poly убит через KILL"
EOF
}

remote_start_poly() {
  # shellcheck disable=SC2029
  ssh "$REMOTE" "bash -s" -- "$REMOTE_DIR" <<'EOF'
set -euo pipefail
D="${1:-}"
D="${D%/}"
if [[ -z "$D" || ! -d "$D" ]]; then
  echo "[deploy/remote] start failed: нет каталога $D" >&2
  exit 1
fi
cd "$D"
chmod +x poly
rm -f nohup-poly.log
LD_LIBRARY_PATH="$D:${LD_LIBRARY_PATH:-}" nohup ./poly >> nohup-poly.log 2>&1 < /dev/null &
pid=$!
echo "[deploy/remote] started ./poly pid=$pid"
EOF
}

cd "$REPO_ROOT"
# Сервер без GPU: prebuilt libxgboost (~секунды), без CUDA, без cmake-сборки.
cargo build --no-default-features --features xgb-prebuilt --profile release
cp "$REPO_ROOT/target/release/poly" "$SCRIPT_DIR/poly"
cd "$SCRIPT_DIR"

echo "[deploy] Останавливаем существующий poly на $REMOTE перед rsync…"
remote_stop_poly

rsync -avz ./ "$REMOTE:$REMOTE_DIR/"

echo "[deploy] Контрольная остановка poly на $REMOTE после rsync…"
remote_stop_poly

echo "[deploy] Старт ./poly на $REMOTE (nohup)…"
remote_start_poly

echo "[deploy] Готово. Лог: $REMOTE_DIR/nohup-poly.log  →  ssh $REMOTE 'tail -f $REMOTE_DIR/nohup-poly.log'"
