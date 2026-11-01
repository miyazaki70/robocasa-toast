"""RoboCasa の環境を lerobot のラッパー経由で作り、初期フレームを保存して数ステップ動かす。

lerobot-eval と同じ RoboCasaEnv を使うので、ここで通れば評価でも環境は作れる。
  cd /root && python $WORK/tools/robocasa/robocasa_smoke_test.py --task GetToastedBread --out $OUTPUTS/robocasa_smoke
（/content で実行すると clone した robocasa ディレクトリが誤って import されるので、別の場所で実行する）
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.envs.robocasa import RoboCasaEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="GetToastedBread")
    ap.add_argument("--split", default="target")
    ap.add_argument("--obj-registries", nargs="+", default=["aigen", "lightwheel"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = RoboCasaEnv(task=args.task, split=args.split, obj_registries=args.obj_registries)
    print(f"horizon: {env._max_episode_steps} steps ({env._max_episode_steps / env.metadata['render_fps']:.0f} s)")

    for seed in args.seeds:
        t0 = time.time()
        obs, _ = env.reset(seed=seed)
        t_reset = time.time() - t0
        print(f"\n[seed {seed}] reset {t_reset:.1f}s")
        print(f"  instruction: {env.task_description!r}")
        print(f"  layout/style: {env._env.env.layout_id} / {env._env.env.style_id}")
        print(f"  objects: {sorted(env._env.env.objects)}")
        print(f"  agent_pos: {obs['agent_pos'].shape} {np.round(obs['agent_pos'], 3)}")
        frames = [obs["pixels"][cam] for cam in env.camera_name]
        for cam, img in zip(env.camera_name, frames):
            print(f"  {cam}: {img.shape} {img.dtype}")
        Image.fromarray(np.concatenate(frames, axis=1)).save(out / f"{args.task}_seed{seed}.png")

        t0 = time.time()
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
        dt = time.time() - t0
        print(f"  {args.steps} zero-action steps: {dt:.1f}s ({args.steps / dt:.1f} steps/s), success={info['is_success']}")

    env.close()
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
