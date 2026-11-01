"""GetToastedBread の評価で、どのサブタスクまで進んだかを記録する。

lerobot-eval をそのまま実行し、RoboCasaGymEnv.step だけを包んで毎 step 環境の内部状態を記録する
（--eval.batch_size=1 なら SyncVectorEnv で同じプロセス内に環境が作られるので、差し替えが効く）。
引数は lerobot-eval と同じ。結果は --output_dir の diag_steps.csv / diag_episodes.json と標準出力の表。
動画は lerobot-eval の既定（先頭 10 本）と違い、全エピソード分を videos/ に保存する。

  cd /root && python $WORK/tools/robocasa/diagnose_toast_eval.py --policy.path=... --env.type=robocasa \
    --env.task=GetToastedBread ... --output_dir=$OUTPUTS/robocasa_toast_ft_010000_diag

判定の段階（robocasa get_toasted_bread.py の _check_success と Toaster.update_state に対応）:
  1. lever_down : いずれかのスロットのレバーが 0.90 以上（トースターが作動する値）
  2. toaster_on : パンがスロット床に接したままトースターが作動した（_check_success 内のラッチ）
  3. popped     : 作動から 500 step 経ってレバーが戻った（cooldown に入った）
  4. grasped    : パンを掴んだ（OU.check_obj_grasped）
  5. lifted     : パンが初期位置から 5cm 以上持ち上がった
  6. in_plate   : パンが皿に接し、水平距離が閾値内（OU.check_obj_in_receptacle）
  7. success    : toaster_on かつ in_plate かつ グリッパーがパンから 25cm 以上離れた
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from toast_task import STAGES

LIFT_TH = 0.05


def _arg(name):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return None


OUT_DIR = None  # __main__ で --output_dir から決める（集計だけ import するときは使わない）
# 並列実行（diagnose_toast_parallel.py）でエピソードを分担するとき、各プロセスの先頭エピソードの通し番号
EP_OFFSET = int(os.environ.get("DIAG_EPISODE_OFFSET", "0"))
# 0 より大きければ、初めて成功した後もこの step 数だけエピソードを続ける（既定 0 は成功した step で終わる）
HOLD_AFTER_SUCCESS = int(os.environ.get("DIAG_HOLD_AFTER_SUCCESS", "0"))
# 指定すると、各エピソードの開始時に torch の乱数（方策のノイズ）をこの値＋エピソード番号で初期化する。
# 配置（--seed）を変えずに方策の試行だけを変えたいとき用。GPU の計算は完全には決定的でないので、同じ値でも同じ動きになるとは限らない
POLICY_SEED = os.environ.get("DIAG_POLICY_SEED")

steps_rows = []
episodes = []
cur = None


def _new_episode(kenv, seed):
    bread_pos = np.array(kenv.sim.data.body_xpos[kenv.obj_body_id["obj"]])
    toaster = kenv.toaster
    return {
        "episode": EP_OFFSET + len(episodes),
        "seed": seed,
        "layout_id": int(getattr(kenv, "layout_id", -1)),
        "style_id": int(getattr(kenv, "style_id", -1)),
        "toaster_xml": str(getattr(toaster, "xml", getattr(toaster, "name", ""))),
        "n_slot_pairs": len(toaster._slot_pairs),
        "bread_z0": float(bread_pos[2]),
        "bread_xy0": bread_pos[:2].tolist(),
        "n_steps": 0,
        "max_lever": 0.0,
        "first": {},  # 各段階に最初に到達した step
        "max_lift": 0.0,
        "min_bread_plate_xy": float("inf"),
        "min_eef_bread": float("inf"),
        "min_eef_toaster": float("inf"),
        "bread_slot_contact_start": None,
        "bread_slot_contact_end": None,
        "success": False,
    }


def _record(kenv, success):
    from robocasa.utils import object_utils as OU

    ep = cur
    t = ep["n_steps"]
    ep["n_steps"] += 1
    toaster = kenv.toaster
    st = toaster.get_state(kenv)
    levers = [float(st[sp]["lever"]) for sp in toaster._slot_pairs]
    turned_on = [bool(st[sp]["turned_on"]) for sp in toaster._slot_pairs]
    cooldown = [int(toaster._cooldown[sp]) for sp in toaster._slot_pairs]
    num_on = [int(toaster._num_steps_on[sp]) for sp in toaster._slot_pairs]
    slot_contact = bool(toaster.check_slot_contact(kenv, "obj"))

    bread = np.array(kenv.sim.data.body_xpos[kenv.obj_body_id["obj"]])
    plate = np.array(kenv.sim.data.body_xpos[kenv.obj_body_id["plate"]])
    eef = np.array(kenv.sim.data.site_xpos[kenv.robots[0].eef_site_id["right"]])
    toaster_pos = np.array(kenv.sim.data.get_body_xpos(toaster.root_body))
    lift = float(bread[2] - ep["bread_z0"])
    bread_plate_xy = float(np.linalg.norm(bread[:2] - plate[:2]))
    eef_bread = float(np.linalg.norm(eef - bread))
    eef_toaster = float(np.linalg.norm(eef - toaster_pos))
    grasped = bool(OU.check_obj_grasped(kenv, "obj"))
    in_plate = bool(OU.check_obj_in_receptacle(kenv, "obj", "plate"))
    far = bool(OU.gripper_obj_far(kenv, "obj"))
    toaster_on = bool(getattr(kenv, "toaster_on", False))

    if t == 0:
        ep["bread_slot_contact_start"] = slot_contact
    ep["bread_slot_contact_end"] = slot_contact
    ep["max_lever"] = max(ep["max_lever"], max(levers))
    ep["max_lift"] = max(ep["max_lift"], lift)
    ep["min_bread_plate_xy"] = min(ep["min_bread_plate_xy"], bread_plate_xy)
    ep["min_eef_bread"] = min(ep["min_eef_bread"], eef_bread)
    ep["min_eef_toaster"] = min(ep["min_eef_toaster"], eef_toaster)
    stages = {
        "lever_down": max(levers) >= 0.90,
        "turned_on": any(turned_on),
        "toaster_on": toaster_on,
        "popped": any(c > 0 for c in cooldown),
        "grasped": grasped,
        "lifted": lift >= LIFT_TH,
        "in_plate": in_plate,
        "success": bool(success),
    }
    for k, v in stages.items():
        if v and k not in ep["first"]:
            ep["first"][k] = t
    ep["success"] = ep["success"] or bool(success)

    steps_rows.append(
        {
            "episode": ep["episode"],
            "step": t,
            "levers": ";".join(f"{v:.3f}" for v in levers),
            "turned_on": ";".join(str(int(v)) for v in turned_on),
            "num_steps_on": ";".join(str(v) for v in num_on),
            "cooldown": ";".join(str(v) for v in cooldown),
            "toaster_on": int(toaster_on),
            "bread_slot_contact": int(slot_contact),
            "grasped": int(grasped),
            "lift": round(lift, 4),
            "bread_plate_xy": round(bread_plate_xy, 4),
            "in_plate": int(in_plate),
            "gripper_far": int(far),
            "eef_bread": round(eef_bread, 4),
            "eef_toaster": round(eef_toaster, 4),
            "success": int(bool(success)),
        }
    )


def _finish():
    global cur
    if cur is not None and cur["n_steps"] > 0:
        episodes.append(cur)
        e = cur
        print(
            f"[diag] ep{e['episode']:02d} steps={e['n_steps']} max_lever={e['max_lever']:.2f} "
            f"first={e['first']} max_lift={e['max_lift']:.3f} "
            f"min_bread_plate_xy={e['min_bread_plate_xy']:.2f} min_eef_bread={e['min_eef_bread']:.3f}",
            flush=True,
        )
        _dump()
    cur = None


def _dump():
    with open(OUT_DIR / "diag_episodes.json", "w") as f:
        json.dump(episodes, f, indent=1)
    if steps_rows:
        with open(OUT_DIR / "diag_steps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(steps_rows[0].keys()))
            w.writeheader()
            w.writerows(steps_rows)


def _summary(episodes):
    n = len(episodes)
    print(f"\n[diag] 段階ごとの到達エピソード数（全 {n} エピソード）")
    for s in STAGES:
        hit = [e for e in episodes if s in e["first"]]
        steps = [e["first"][s] for e in hit]
        med = f"  初到達 step 中央値 {int(np.median(steps))}" if steps else ""
        print(f"  {s:11s} {len(hit):2d}/{n}{med}")
    print(f"  レバー最大値の分布: {[round(e['max_lever'], 2) for e in episodes]}")


def install():
    from robocasa.wrappers import gym_wrapper

    orig_reset = gym_wrapper.RoboCasaGymEnv.reset
    orig_step = gym_wrapper.RoboCasaGymEnv.step

    def reset(self, seed=None, options=None):
        global cur
        _finish()
        out = orig_reset(self, seed=seed, options=options)
        if POLICY_SEED is not None:
            import torch

            torch.manual_seed(int(POLICY_SEED) + len(episodes))
        cur = _new_episode(self.env, seed)
        return out

    def step(self, action_dict):
        out = orig_step(self, action_dict)
        if cur is not None:
            success = out[4].get("success", False)
            _record(self.env, success)
            # 成功してもすぐ終わらせず、HOLD_AFTER_SUCCESS step 動かし続けてから成功として終える（置いた後の動画を見るため）。
            # 記録（diag_steps.csv の success）は毎 step の実際の判定のまま
            if HOLD_AFTER_SUCCESS > 0 and "success" in cur["first"]:
                out[4]["success"] = cur["n_steps"] - 1 - cur["first"]["success"] >= HOLD_AFTER_SUCCESS
        return out

    gym_wrapper.RoboCasaGymEnv.reset = reset
    gym_wrapper.RoboCasaGymEnv.step = step

    # lerobot-eval は録画本数を 10 に固定しているので、全エピソードを録画するよう差し替える
    from lerobot.scripts import lerobot_eval

    orig_eval_policy_all = lerobot_eval.eval_policy_all

    def eval_policy_all(*a, **k):
        if k.get("max_episodes_rendered", 0) > 0:
            k["max_episodes_rendered"] = k["n_episodes"]
        return orig_eval_policy_all(*a, **k)

    lerobot_eval.eval_policy_all = eval_policy_all


if __name__ == "__main__":
    OUT_DIR = Path(_arg("output_dir"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    install()
    from lerobot.scripts.lerobot_eval import main

    try:
        main()
    finally:
        _finish()
        _dump()
        _summary(episodes)
