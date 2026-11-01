"""diagnose_toast_parallel.py の出力（diag_episodes.json と diag_steps.csv）から、段階ごとの到達本数を表にする。

docs/PROGRESS.md の比較表と同じ列に加えて、「ポップアップ後に掴めた本数」を出す。
記憶のないモデルはポップアップ後のトースターをレバーを押す前と区別できず、掴めないことが多かったので、これを主な指標にしている。

  python3 $WORK/tools/robocasa/summarize_toast_diag.py $OUTPUTS/robocasa_toast_mem_ft_*_diag50*
"""

import argparse
import csv
import json
from pathlib import Path

COLUMNS = [("toaster_on", "レバー作動"), ("popped", "ポップアップ"), ("grasped", "掴む"),
           ("lifted", "持ち上げる"), ("in_plate", "皿の上"), ("success", "成功")]


def grasped_after_pop(out: Path, popped: dict[int, int]) -> list[int]:
    """ポップアップした step 以降にパンを掴んだエピソード。"""
    hit = set()
    with open(out / "diag_steps.csv") as f:
        for r in csv.DictReader(f):
            ep = int(r["episode"])
            if ep in popped and ep not in hit and int(r["step"]) >= popped[ep] and r["grasped"] in ("1", "True"):
                hit.add(ep)
    return sorted(hit)


def summarize(out: Path) -> dict:
    episodes = json.loads((out / "diag_episodes.json").read_text())
    row = {"name": out.name, "n": len(episodes)}
    for key, _ in COLUMNS:
        row[key] = sum(key in e["first"] for e in episodes)
    popped = {e["episode"]: e["first"]["popped"] for e in episodes if "popped" in e["first"]}
    row["grasped_after_pop"] = grasped_after_pop(out, popped)
    row["success_eps"] = [e["episode"] for e in episodes if "success" in e["first"]]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    args = ap.parse_args()

    print("| 評価 | 本数 | " + " | ".join(label for _, label in COLUMNS) + " | ポップアップ後に掴む |")
    print("|---" * (len(COLUMNS) + 3) + "|")
    for d in args.dirs:
        if not (d / "diag_episodes.json").exists():
            continue
        r = summarize(d)
        gap = r["grasped_after_pop"]
        print(f"| {r['name']} | {r['n']} | " + " | ".join(str(r[k]) for k, _ in COLUMNS)
              + f" | {len(gap)}/{r['popped']}" + (f"（ep {', '.join(map(str, gap))}）" if gap else "") + " |")


if __name__ == "__main__":
    main()
