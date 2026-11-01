#!/bin/bash
# lerobot を固定コミットで $REPO に clone し、venv（$VENV、ローカルディスク）を作る。ランタイムを起動するたびに実行する（約 15 分）。
#   bash $WORK/scripts/setup_lerobot.sh            # pi + training の extra（引数を渡すとその extra を入れる）
# lerobot 本体はこのリポジトリでは管理しない（変更は加えていない）。$REPO が既にあれば clone はしない。
set -euo pipefail
source "$(dirname "$0")/env.sh"

LEROBOT_SHA=146d8474932c8b566717eb2738ba459337ab71c4   # 2026-10-17 時点の main。これまでの実験はすべてこのコミット
EXTRAS=("$@")
[ ${#EXTRAS[@]} -gt 0 ] || EXTRAS=(pi training)

if [ ! -d "$REPO/.git" ]; then
  git clone -q https://github.com/huggingface/lerobot.git "$REPO"
  git -C "$REPO" checkout -q $LEROBOT_SHA
fi
[ "$(git -C "$REPO" rev-parse HEAD)" = $LEROBOT_SHA ] || echo "[注意] $REPO は $LEROBOT_SHA ではない（$(git -C "$REPO" rev-parse --short HEAD)）"

command -v uv > /dev/null || pip install -q uv
cd "$REPO"
uv sync --locked $(printf -- '--extra %s ' "${EXTRAS[@]}")
python -c "import lerobot; print('lerobot ok', lerobot.__file__)"
