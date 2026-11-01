#!/bin/bash
# train_robocasa_toast.sh のチェックポイントを、サブタスクの到達を記録しながら評価する（diagnose_toast_parallel.py）。
# チェックポイントがまだなければ保存されるまで待つ。
#   bash $WORK/scripts/eval_robocasa_toast.sh <base|mem> <checkpoint の step（例 004000）> [並列数=1] [本数=50]
# 並列数の目安: 学習中は 1（mem の学習は GPU を約 79GB 使う）、学習が止まっていれば 5（1 プロセス GPU 約 11GB）。
# seed は 1000〜（本数-1）。これまでの評価と同じ配置。lerobot-eval の引数を足したいときは EVAL_EXTRA に入れる
#   （例 EVAL_EXTRA=--env.episode_length=3600。出力名は OUT_SUFFIX=_h3600 のように変える）。
# 出力 $OUTPUTS/<EVAL_PREFIX>_<更新回数>_diag<本数><OUT_SUFFIX>/、ログは同名の .log。
# mem は勾配累積 2 なので、step 004000 / 008000 / 012000 が更新 2000 / 4000 / 6000 回（出力名は更新回数）。
set -uo pipefail
source "$(dirname "$0")/common.sh"
load_variant "${1:?variant（base / mem）}" || exit 1

S=${2:?checkpoint の step（例 004000）}
W=${3:-1}
N=${4:-50}
CK=$OUT_DIR/checkpoints
U=$(printf %06d $((10#$S / UPDATE_DIV)))
OUT=$OUTPUTS/${EVAL_PREFIX}_${U}_diag${N}${OUT_SUFFIX:-}

# last がこの step 以降を指せば保存は完了している（lerobot-train は保存の最後に last を付け替える）
until [ -d "$CK/$S/pretrained_model" ] && { [ "$(readlink "$CK/last")" = "$S" ] || [ "$(readlink "$CK/last")" \> "$S" ]; }; do
  train_running || { echo "$CK/$S がなく、学習も動いていない"; exit 1; }
  sleep 60
done

cd /root   # /content だと clone した robocasa が名前空間パッケージとして import される
echo "=== $VARIANT $S (update $U) start $(date -u +%H:%M) workers=$W -> $OUT"
# shellcheck disable=SC2086  # EVAL_EXTRA は空白区切りで複数渡せるように分割する
python "$WORK/tools/robocasa/diagnose_toast_parallel.py" --workers "$W" --n-episodes "$N" --seed 1000 --output-dir "$OUT" -- \
  --policy.path="$CK/$S/pretrained_model" \
  --policy.gradient_checkpointing=false --policy.compile_model=false --policy.device=cuda \
  --env.type=robocasa --env.task=GetToastedBread --env.split=target '--env.obj_registries=["aigen","lightwheel"]' \
  ${EVAL_EXTRA:-} \
  > "$OUT.log" 2>&1
echo "=== $VARIANT $S exit $? $(date -u +%H:%M)"
grep -A10 "^成功" "$OUT.log"
