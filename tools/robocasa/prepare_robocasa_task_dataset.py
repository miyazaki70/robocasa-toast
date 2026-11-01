"""RoboCasa の統合データセットから 1 タスク分のエピソードだけを取得し、task 文字列を評価時の指示文に揃える。

lerobot/robocasa_target_human_unified（50 タスク、2.5 万エピソード、動画 70GB 超）は task 文字列がタスク名
（例 "GetToastedBread"）だが、lerobot-eval では環境が自然文の指示を渡す。学習と評価で言語入力が食い違うので、
ローカルに取得したコピーの meta/tasks.parquet と meta/episodes の tasks 列を指示文に書き換える。
データ本体は変えない（task_index はそのまま）。統計量（meta/stats.json）は 50 タスク全体のもののまま
（= ゼロショット用チェックポイント pi05_base_robocasa_step0 と同じ正規化）。

  cd /root && python $WORK/tools/robocasa/prepare_robocasa_task_dataset.py --root $HF_LEROBOT_HOME/robocasa_toast
（--task の既定は GetToastedBread。ほかのタスクでは --instruction も渡す）
出力: <root>/（LeRobotDataset、該当ファイルのみ）と <root>/episodes_<task>.json（学習の --dataset.episodes に渡す）
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from toast_task import HUB_REPO, INSTRUCTION, TASK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=HUB_REPO)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--instruction", help=f"lerobot-eval で環境が渡す指示文（robocasa の get_ep_meta()['lang']）。{TASK} なら省略可")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    if args.instruction is None:
        if args.task != TASK:
            ap.error(f"--task {args.task} には --instruction が必要")
        args.instruction = INSTRUCTION
    root = Path(args.root)

    meta = LeRobotDatasetMetadata(args.repo_id, root=root)
    episodes = [int(e["episode_index"]) for e in meta.episodes if args.task in list(e["tasks"])]
    print(f"{args.task}: {len(episodes)} episodes ({episodes[0]}..{episodes[-1]}), "
          f"{sum(meta.episodes[i]['length'] for i in episodes)} frames")

    # episodes を渡すと、その範囲を含む data / video ファイルだけを取得する
    LeRobotDataset(args.repo_id, root=root, episodes=episodes)

    tasks_path = root / "meta/tasks.parquet"
    tasks = pd.read_parquet(tasks_path)
    if args.task in tasks.index:
        tasks = tasks.rename(index={args.task: args.instruction})
        tasks.to_parquet(tasks_path)
    for ep_file in sorted((root / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(ep_file)
        df = table.to_pandas()
        df["tasks"] = df["tasks"].apply(lambda ts: [args.instruction if t == args.task else t for t in ts])
        pq.write_table(pa.Table.from_pandas(df, schema=table.schema, preserve_index=False), ep_file)

    (root / f"episodes_{args.task}.json").write_text(json.dumps(episodes))

    ds = LeRobotDataset(args.repo_id, root=root, episodes=episodes)
    item = ds[0]
    print("check:", len(ds), "frames |", repr(item["task"]))
    print("       " + ", ".join(f"{k}: {tuple(v.shape)}" for k, v in item.items() if hasattr(v, "shape") and v.ndim > 0))


if __name__ == "__main__":
    main()
