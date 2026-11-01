#!/bin/bash
# Pi0.5 を RoboCasa GetToastedBread で学習する（nohup で起動し、切断に備えて出力は Drive）。
#   bash $WORK/scripts/train_robocasa_toast.sh <base|mem>
#   STEPS=20 bash ... で疎通確認、DRY_RUN=1 bash ... でコマンドの表示だけ。
# 前提（variant ごとの違いは scripts/common.sh）:
#   base : $OUTPUTS/pi05_base_robocasa_step0 と $HF_LEROBOT_HOME/robocasa_toast（prepare_robocasa_task_dataset.py）
#   mem  : base の 10000 step と $HF_LEROBOT_HOME/robocasa_toast
# いずれも新しい学習として始める（optimizer と学習率スケジュールは初期化。warmup・decay は steps に合わせて自動で縮む）。
# 途中で止まった学習の再開は scripts/resume_robocasa_toast.sh。
set -euo pipefail
USER_STEPS=${STEPS:-}
source "$(dirname "$0")/common.sh"
load_variant "${1:?variant（base / mem）}"
STEPS=${USER_STEPS:-$STEPS}

[ -e "$OUT_DIR" ] && { echo "$OUT_DIR が既にある。再開なら scripts/resume_robocasa_toast.sh $VARIANT"; exit 1; }
[ -f "$DS_ROOT/meta/info.json" ] || [ "${DRY_RUN:-0}" = 1 ] || { echo "学習データ $DS_ROOT がない"; exit 1; }

ARGS=(
  --policy.path="$INIT"
  --dataset.repo_id="$DS_REPO" --dataset.root="$DS_ROOT"
)
# Hub の 50 タスク統合データから取得したものは、GetToastedBread のエピソードだけを使う
if [ -f "$DS_ROOT/episodes_GetToastedBread.json" ]; then
  ARGS+=("--dataset.episodes=$(tr -d ' ' < "$DS_ROOT/episodes_GetToastedBread.json")")
fi
ARGS+=(
  "${EXTRA_ARGS[@]}"
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=false
  --policy.device=cuda --policy.push_to_hub=false
  --batch_size="$BATCH"
)
[ "$GRAD_ACCUM" -gt 1 ] && ARGS+=(--accelerator.gradient_accumulation.steps="$GRAD_ACCUM")
ARGS+=(
  --steps="$STEPS" --save_freq="$SAVE_FREQ" --log_freq=50 --num_workers=8
  --output_dir="$OUT_DIR" --job_name="$JOB" --wandb.enable=false
)
launch_train "${ARGS[@]}"
