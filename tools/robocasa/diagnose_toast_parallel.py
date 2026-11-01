"""diagnose_toast_eval.py を複数プロセスで並列に回し、結果を 1 つにまとめる。

エピソード i は seed+i で初期化される（batch_size=1 の lerobot-eval と同じ）ので、プロセスごとに
--seed と --eval.n_episodes をずらして分担すれば、逐次実行と同じ配置（レイアウト・物体）を評価できる。
--eval.batch_size>1 は使わない（ワーカーが forkserver で起動して診断の差し替えが効かず、seed も重複する）。

  cd /root && python $WORK/tools/robocasa/diagnose_toast_parallel.py --workers 5 --n-episodes 20 --seed 1000 \
    --output-dir $OUTPUTS/robocasa_toast_ft_010000_diag20 -- \
    --policy.path=... --policy.gradient_checkpointing=false --policy.compile_model=false --policy.device=cuda \
    --env.type=robocasa --env.task=GetToastedBread --env.split=target '--env.obj_registries=["aigen","lightwheel"]'

scripts/eval_robocasa_toast.sh がこれを学習の variant とチェックポイントに合わせて呼ぶ。
1 プロセスあたり GPU 約 11GB・CPU 約 6 コア。出力:
  <output-dir>/shards/shard_e<first>-<last>/  各プロセスの lerobot-eval 出力（shard.json に担当範囲）とログ
  <output-dir>/videos/<task>_0/eval_episode_<通し番号>.mp4
  <output-dir>/diag_episodes.json, diag_steps.csv, eval_summary.json

途中で止まった shard があっても、終わったエピソードはまとめる。止まった分は --episodes a-b で同じ --output-dir に追加すると
自動でまとめ直す（--merge-only でまとめ直しだけもできる）。
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from diagnose_toast_eval import _summary


def merge(out: Path, t0: float | None = None) -> None:
    """shards/*/shard.json のある shard の結果を 1 つにまとめる（終わったエピソードだけ。途中で止まった shard も可）。"""
    episodes, rows, results = [], [], []
    for meta_path in sorted((out / "shards").glob("*/shard.json")):
        sd = meta_path.parent
        meta = json.loads(meta_path.read_text())
        if not (sd / "diag_episodes.json").exists():
            continue
        eps = json.load(open(sd / "diag_episodes.json"))
        done = {e["episode"] for e in eps}
        episodes += eps
        with open(sd / "diag_steps.csv") as f:
            rows += [r for r in csv.DictReader(f) if int(r["episode"]) in done]
        results += [{"episode": e["episode"], "seed": e["seed"], "success": bool(e["success"])} for e in eps]
        for v in sorted(sd.glob("videos/*/eval_episode_*.mp4")):
            ep = meta["start"] + int(v.stem.rsplit("_", 1)[1])
            if ep in done:
                vdst = out / "videos" / v.parent.name
                vdst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(v, vdst / f"eval_episode_{ep}.mp4")

    ids = [e["episode"] for e in episodes]
    if len(ids) != len(set(ids)):
        sys.exit(f"同じエピソードが複数の shard にある: {sorted(i for i in set(ids) if ids.count(i) > 1)}")
    episodes.sort(key=lambda e: e["episode"])
    rows.sort(key=lambda r: (int(r["episode"]), int(r["step"])))
    results.sort(key=lambda r: r["episode"])
    with open(out / "diag_episodes.json", "w") as f:
        json.dump(episodes, f, indent=1)
    with open(out / "diag_steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_success = sum(r["success"] for r in results)
    missing = sorted(set(range(max(ids) + 1)) - set(ids))
    with open(out / "eval_summary.json", "w") as f:
        json.dump({"n_episodes": len(results), "n_success": n_success,
                   "pc_success": 100 * n_success / max(len(results), 1),
                   "wall_s": None if t0 is None else time.time() - t0, "missing_episodes": missing,
                   "episodes": results}, f, indent=1)
    print(f"\n成功 {n_success}/{len(results)}" + ("" if t0 is None else f"、所要 {time.time() - t0:.0f}s"))
    if missing:
        print(f"[注意] 結果のないエピソード: {missing}")
    _summary(episodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--n-episodes", type=int, help="エピソード 0..n-1 を評価する")
    p.add_argument("--episodes", help="評価するエピソードの範囲 'a-b'（途中で止まった分のやり直し用。既存の --output-dir に追加する）")
    p.add_argument("--seed", type=int, required=True, help="エピソード 0 の seed（エピソード i は seed+i）")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--merge-only", action="store_true", help="評価せず、既存の shard の結果をまとめ直す")
    p.add_argument("eval_args", nargs=argparse.REMAINDER, help="-- の後に lerobot-eval の引数")
    args = p.parse_args()
    out = args.output_dir
    if args.merge_only:
        return merge(out)
    eval_args = [a for a in args.eval_args if a != "--"]
    for a in eval_args:
        if a.split("=")[0] in ("--seed", "--eval.n_episodes", "--eval.batch_size", "--output_dir", "--job_name"):
            sys.exit(f"{a} はこのスクリプトが決めるので渡さない")

    if args.episodes:
        first, last = map(int, args.episodes.split("-"))
    elif args.n_episodes:
        if out.exists():
            sys.exit(f"{out} が既にある（追加で回すなら --episodes を使う）")
        first, last = 0, args.n_episodes - 1
    else:
        sys.exit("--n-episodes か --episodes を指定する")
    (out / "shards").mkdir(parents=True, exist_ok=True)

    # エピソードをなるべく均等に分ける（例: 20 を 3 → 7, 7, 6）
    base, extra = divmod(last - first + 1, args.workers)
    shards, start = [], first
    for k in range(args.workers):
        n = base + (1 if k < extra else 0)
        if n:
            shards.append((start, n))
        start += n

    diag = Path(__file__).resolve().parent / "diagnose_toast_eval.py"
    procs = []
    t0 = time.time()
    for start, n in shards:
        name = f"shard_e{start:03d}-{start + n - 1:03d}"
        sd = out / "shards" / name
        if sd.exists():
            sys.exit(f"{sd} が既にある")
        sd.mkdir(parents=True)
        (sd / "shard.json").write_text(json.dumps({"start": start, "n": n, "seed": args.seed + start}))
        cmd = [
            sys.executable, str(diag), *eval_args,
            f"--seed={args.seed + start}", f"--eval.n_episodes={n}", "--eval.batch_size=1",
            f"--output_dir={sd}", f"--job_name=diag_{name}",
        ]
        env = {**os.environ, "DIAG_EPISODE_OFFSET": str(start)}
        log = open(out / "shards" / f"{name}.log", "w")
        procs.append((name, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
        print(f"{name}: episodes {start}..{start + n - 1} (seed {args.seed + start}) pid {procs[-1][1].pid}", flush=True)

    failed = []
    for name, proc in procs:
        if proc.wait() != 0:
            failed.append(name)
        print(f"{name} 終了 (exit {proc.returncode}, {time.time() - t0:.0f}s)", flush=True)
    merge(out, t0)
    if failed:
        sys.exit(f"失敗した shard: {failed}（{out}/shards/<shard>.log を見る。--episodes で止まった分だけやり直せる）")


if __name__ == "__main__":
    main()
