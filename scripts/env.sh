# 作業用の環境変数
#   source <このリポジトリ>/scripts/env.sh
# WORK はこのファイルの場所から決まる（Drive 以外に clone しても動く）。上書きしたいときは先に export しておく。
export WORK=${WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export REPO=${REPO:-$WORK/lerobot}              # lerobot の clone（scripts/setup_lerobot.sh）。Git 管理外
export OUTPUTS=${OUTPUTS:-$REPO/outputs}        # 学習・評価の出力（lerobot の .gitignore 対象）
export VENV=${VENV:-/content/lerobot-venv}      # Drive は I/O が遅いので venv はローカルディスク（セッションごとに再作成）
export UV_PROJECT_ENVIRONMENT=$VENV
case ":$PATH:" in *":$VENV/bin:"*) ;; *) export PATH=$VENV/bin:$PATH ;; esac
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Colab は glvnd に NVIDIA の EGL が登録されておらず、そのままだと Mesa llvmpipe（CPU）で描画される。
# RoboCasa では 3 カメラの描画が 1 step 約 100ms → 1ms になる（2026-10-19 計測）。
if [ -f /usr/lib64-nvidia/libEGL_nvidia.so.0 ]; then
  mkdir -p /content/egl_vendor
  printf '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib64-nvidia/libEGL_nvidia.so.0"}}\n' > /content/egl_vendor/10_nvidia.json
  export __EGL_VENDOR_LIBRARY_FILENAMES=/content/egl_vendor/10_nvidia.json
fi
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/content/hf_lerobot}
export HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}  # 先に `hf auth login` しておく
