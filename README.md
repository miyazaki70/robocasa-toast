# robocasa-toast

Pi0.5（[LeRobot](https://github.com/huggingface/lerobot)）を、シミュレーション環境 RoboCasa のタスク GetToastedBread（トースターでパンを焼いて食卓の皿まで運ぶ）で追加学習・評価した作業の記録とスクリプト。
動作確認環境: Google Colab（NVIDIA RTX PRO 6000 Blackwell 96GB）。

| 学習 | 内容 | 結果（50 本） |
|---|---|---|
| 基本学習（`base`） | `lerobot/pi05_base` に RoboCasa の入出力を付け、GetToastedBread のデモ 506 本で 10000 step | 成功 0。パンは掴めるが、レバーを押し切れず「押す → 待つ → 取り出す」の順序も守れない |
| 視覚記憶（`mem`） | 上を起点に、MEM の視覚記憶（1 秒おきに過去 5 秒分の画像）を有効にして 4000 回更新 | 成功 1（ep 36 が 1,803 step で全工程を完了。上限 3600 step で評価）。7 本がポップアップを待ってからパンを取り出した |

レポートの「A. レバーのデモを追加」（TurnOnToaster のデモ 2,508 本を追加。成功せず）は、このリポジトリのスクリプトには含まない。

まとめは [`docs/report/report.pdf`](docs/report/report.pdf)（レポート）。詳細は [`docs/PROGRESS.md`](docs/PROGRESS.md)（経緯と結果）。

## 構成

```
scripts/                 シェルのエントリポイント（どれも scripts/env.sh を読む）
  env.sh                 環境変数（WORK はこのリポジトリの場所から自動で決まる）
  setup_lerobot.sh       lerobot を固定コミットで clone して venv を作る
  setup_robocasa.sh      RoboCasa / robosuite とアセットを入れる
  common.sh              学習 2 種類（base / mem）の設定と共通関数
  train_robocasa_toast.sh  <variant>                     学習の起動
  resume_robocasa_toast.sh <variant>                     checkpoints/last から再開
  eval_robocasa_toast.sh   <variant> <step> [並列数] [本数]  評価（サブタスクの到達を記録）
tools/
  robocasa/              データ準備・起点チェックポイント作成・診断付き評価・集計
docs/
  PROGRESS.md            作業記録（経緯と結果）
  report/                レポート（report.md が元。bash docs/report/build.sh で report.html と report.pdf を作る）
```

Git で管理しないもの（`.gitignore`）: `lerobot/`（上流の clone。変更なし）、`datasets/`、チェックポイント・動画・ログなどの大きなファイル。
学習・評価の出力は `lerobot/outputs/`（`$OUTPUTS`）に置く。

## 使い方

```bash
hf auth login                                  # google/paligemma-3b-pt-224 の利用承認済みアカウント
source <このリポジトリ>/scripts/env.sh
bash $WORK/scripts/setup_lerobot.sh            # 約 15 分
bash $WORK/scripts/setup_robocasa.sh           # RoboCasa の評価に必要（アセット約 18GB）
```

学習データと起点チェックポイント:

```bash
# 学習データ（GetToastedBread の 506 本、8.0GB）と起点チェックポイント
python $WORK/tools/robocasa/prepare_robocasa_task_dataset.py --root $HF_LEROBOT_HOME/robocasa_toast
python $WORK/tools/robocasa/make_pi05_robocasa_step0.py --out $OUTPUTS/pi05_base_robocasa_step0
```

基本学習（`base`）:

```bash
# 学習（DRY_RUN=1 でコマンドの確認だけ、STEPS=20 で疎通確認）
bash $WORK/scripts/train_robocasa_toast.sh base

# 評価（50 本、5 並列）と集計
bash $WORK/scripts/eval_robocasa_toast.sh base 010000 5
python3 $WORK/tools/robocasa/summarize_toast_diag.py $OUTPUTS/robocasa_toast_*_diag50*
```

視覚記憶（`mem`、基本学習の 10000 step が起点）:

```bash
bash $WORK/scripts/train_robocasa_toast.sh mem

# 4000 回更新（勾配累積 2 なので step 008000）を、上限 3600 step で評価
EVAL_EXTRA=--env.episode_length=3600 OUT_SUFFIX=_h3600 bash $WORK/scripts/eval_robocasa_toast.sh mem 008000 5
python3 $WORK/tools/robocasa/summarize_toast_diag.py $OUTPUTS/robocasa_toast_mem_ft_*_diag50*
```

- 学習のチェックポイントは 1 個 23GB（`pretrained_model` 9GB ＋ `training_state` 15GB）。最新以外の `training_state` は学習中に自動で消す。
- `mem` の学習は GPU メモリを約 79GB 使う（batch 16 × 勾配累積 2）。
