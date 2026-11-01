# train / resume / eval の各スクリプトが共有する設定と関数。単独では実行しない（source して使う）。
#
# RoboCasa GetToastedBread の学習は 2 種類（variant）ある。経緯は docs/PROGRESS.md。
#   base  : pi05_base_robocasa_step0 から GetToastedBread 506 本で 10000 step（3 章）
#   mem   : base の 10000 step を起点に、MEM の視覚記憶を有効にして 6000 回更新（12000 micro-step、4.2）
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

STEP0=$OUTPUTS/pi05_base_robocasa_step0                                      # tools/robocasa/make_pi05_robocasa_step0.py
BASE_10K=$OUTPUTS/train/pi05_robocasa_toast/checkpoints/010000/pretrained_model

# variant ごとの設定を変数に入れる。
#   JOB, INIT, DS_REPO, DS_ROOT, STEPS, SAVE_FREQ, BATCH, GRAD_ACCUM
#   EXTRA_ARGS（lerobot-train に足す引数）, EXTRA_ENV（lerobot-train に渡す環境変数）
#   EVAL_PREFIX（評価の出力名）, UPDATE_DIV（checkpoint の step を更新回数に直す割り算。勾配累積の回数）
#   OUT_DIR, LOG, PID_FILE
load_variant() {
  VARIANT=${1:?variant（base / mem）}
  EXTRA_ARGS=()
  EXTRA_ENV=()
  GRAD_ACCUM=1
  case $VARIANT in
    base)
      JOB=pi05_robocasa_toast
      INIT=$STEP0
      DS_REPO=lerobot/robocasa_target_human_unified
      DS_ROOT=$HF_LEROBOT_HOME/robocasa_toast
      STEPS=10000 SAVE_FREQ=2000 BATCH=32
      EVAL_PREFIX=robocasa_toast_ft
      ;;
    mem)
      # 視覚記憶: 1 秒（20 フレーム）おきに 6 枚（現在＋過去 5 秒）を SigLIP の中で時間方向に混ぜる。追加パラメータはない。
      # batch 32 は GPU メモリ（95GB）に収まらないので、batch 16 × 勾配累積 2（実効 32）。
      # lerobot-train の step は micro-batch を数えるので、12000 step で 6000 回の更新、4000 ごとに保存（2000 更新ごと）。
      JOB=pi05_robocasa_toast_mem
      INIT=$BASE_10K
      DS_REPO=lerobot/robocasa_target_human_unified
      DS_ROOT=$HF_LEROBOT_HOME/robocasa_toast
      STEPS=12000 SAVE_FREQ=4000 BATCH=16 GRAD_ACCUM=2
      EXTRA_ARGS=(--policy.use_visual_memory=true --policy.memory_frames=6 --policy.memory_stride=20)
      EXTRA_ENV=(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
      EVAL_PREFIX=robocasa_toast_mem_ft
      ;;
    *)
      echo "不明な variant: $VARIANT（base / mem）" >&2
      return 1
      ;;
  esac
  UPDATE_DIV=$GRAD_ACCUM
  OUT_DIR=$OUTPUTS/train/$JOB
  LOG=$OUTPUTS/train_$JOB.log
  PID_FILE=/content/train_$VARIANT.pid
}

# PID_FILE の学習プロセスが動いているか
train_running() {
  [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

# 学習が動いている間、最新以外のチェックポイントから training_state/（約 15GB、--resume にだけ必要）を消して Drive を節約する
start_pruner() {
  nohup bash -c "
    while kill -0 \$(cat $PID_FILE) 2>/dev/null; do
      sleep 300
      ls -d $OUT_DIR/checkpoints/[0-9]* 2>/dev/null | sort | head -n -1 | while read d; do rm -rf \$d/training_state; done
    done" > /dev/null 2>&1 &
}

# lerobot-train を nohup で起動し、PID を PID_FILE に書いて pruner を付ける。引数は lerobot-train にそのまま渡す。
# DRY_RUN=1 なら起動せずにコマンドを表示するだけ。
launch_train() {
  local redirect=${LOG_MODE:->}
  if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "[dry-run] cd /root && env ${EXTRA_ENV[*]} lerobot-train $* $redirect $LOG"
    return 0
  fi
  cd /root   # /content だと clone した robocasa が名前空間パッケージとして import される
  if [ "$redirect" = ">>" ]; then
    env "${EXTRA_ENV[@]}" nohup lerobot-train "$@" >> "$LOG" 2>&1 &
  else
    env "${EXTRA_ENV[@]}" nohup lerobot-train "$@" > "$LOG" 2>&1 &
  fi
  echo $! > "$PID_FILE"
  echo "train pid $(cat "$PID_FILE") -> $LOG"
  start_pruner
}
