#!/bin/bash
# RoboCasa365 を lerobot の venv に入れる（セッションごとに、scripts/setup_lerobot.sh の後で実行）。
#   bash $WORK/scripts/setup_lerobot.sh
#   bash $WORK/scripts/setup_robocasa.sh
#
# lerobot の CI（lerobot/docker/Dockerfile.benchmark.robocasa）と同じコミットに固定する。
# 依存は venv にあるもの（scipy, opencv-python-headless, gymnasium など）を活かし、足りないものだけ足す。
# robocasa は numpy==2.2.5 と mujoco==3.3.1 を import 時に assert するので、この 2 つは必ず合わせる。
set -euo pipefail
source "$(dirname "$0")/env.sh"

ROBOCASA_SHA=56e355ccc64389dfc1b8a61a33b9127b975ba681
ROBOSUITE_SHA=aaa8b9b214ce8e77e82926d677b4d61d55e577ab
SRC=/content

[ -d $SRC/robocasa ] || { git clone -q https://github.com/robocasa/robocasa.git $SRC/robocasa && git -C $SRC/robocasa checkout -q $ROBOCASA_SHA; }
[ -d $SRC/robosuite ] || { git clone -q https://github.com/ARISE-Initiative/robosuite.git $SRC/robosuite && git -C $SRC/robosuite checkout -q $ROBOSUITE_SHA; }

# robocasa の setup.py は lerobot==0.3.3 を要求するので --no-deps（robosuite も opencv-python を引くので同様）
uv pip install --python $VENV/bin/python --no-deps -e $SRC/robosuite -e $SRC/robocasa
uv pip install --python $VENV/bin/python \
  "numpy==2.2.5" "numba==0.61.2" "mujoco==3.3.1" "pygame==2.6.1" "pynput==1.8.1" \
  "imageio==2.37.3" "h5py==3.16.0" "lxml==6.0.4" "hidapi==0.14.0.post4" "qpsolvers[quadprog]>=4.3.1"
uv pip install --python $VENV/bin/python --no-deps "tianshou==0.4.10"
# 参考評価で使う lerobot/smolvla_robocasa に必要（lerobot の smolvla extra と同じ指定）
uv pip install --python $VENV/bin/python "num2words>=0.5.14,<0.6.0"

# /content で実行すると clone したディレクトリが名前空間パッケージとして import されるので、別の場所で実行する
cd /root
python -m robosuite.scripts.setup_macros > /dev/null
python -m robocasa.scripts.setup_macros > /dev/null

# 約 18GB（aigen が 13GB）を /content に展開する。Box からの取得は 2026-10-19 時点で約 2 分。
# tex_generative は fixture の XML から無条件に参照されるので必須。
# GetToastedBread の皿（plate）は aigen / objaverse にしかないので objs_aigen も入れる。
python -m robocasa.scripts.download_kitchen_assets --type tex tex_generative fixtures_lw objs_lw objs_aigen < <(yes y)
python -c "import robocasa, robosuite, numpy, mujoco; print('robocasa ok | numpy', numpy.__version__, '| mujoco', mujoco.__version__)"
