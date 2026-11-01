#!/bin/bash
# train_robocasa_toast.sh の学習を、最後に保存したチェックポイント（checkpoints/last）から --resume で続ける。
#   bash $WORK/scripts/resume_robocasa_toast.sh <base|mem>
# 前提: uv sync 済み、学習データが $DS_ROOT にある。
# 設定（batch・勾配累積・step 数・データの場所など）は checkpoints/last/pretrained_model/train_config.json から読まれる。
# 再開には checkpoints/last/training_state/ が必要（optimizer・学習率スケジュール・乱数・step 数）。
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_variant "${1:?variant（base / mem）}"

CFG=$OUT_DIR/checkpoints/last/pretrained_model/train_config.json

[ -d "$OUT_DIR/checkpoints/last/training_state" ] || { echo "$OUT_DIR/checkpoints/last/training_state がない（再開できない）"; exit 1; }
[ -f "$DS_ROOT/meta/info.json" ] || { echo "学習データ $DS_ROOT がない"; exit 1; }
if train_running; then
  echo "学習が既に動いている (pid $(cat "$PID_FILE"))"; exit 1
fi

[ "${DRY_RUN:-0}" = 1 ] || echo "=== resume from $(readlink "$OUT_DIR/checkpoints/last") at $(date -u '+%F %H:%M') UTC" >> "$LOG"
LOG_MODE='>>' launch_train --resume=true --config_path="$CFG"
