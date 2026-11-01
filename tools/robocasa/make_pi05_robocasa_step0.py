"""pi05_base に RoboCasa の入出力と正規化の統計量を付けた「学習 0 step」のチェックポイントを作る。

pi05_base は行動・状態が 32 次元でカメラ名も RoboCasa と違い、正規化の統計量も持たないので、
そのままでは lerobot-eval で RoboCasa を評価できない。lerobot-train に
`--policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base` を渡したときと同じ手順で
policy と processor を組み立て、学習せずに保存する（= ゼロショット評価用。ファインチューニングの起点にもする）。

  python $WORK/tools/robocasa/make_pi05_robocasa_step0.py --out $OUTPUTS/pi05_base_robocasa_step0
"""

import argparse
from pathlib import Path

import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.utils.feature_utils import dataset_to_policy_features
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from toast_task import CAMERAS, HUB_REPO  # CAMERAS: RoboCasa の 3 台を pi05_base の base / left_wrist / right_wrist の順に対応させる


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lerobot/pi05_base")
    ap.add_argument("--dataset", default=HUB_REPO)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds_meta = LeRobotDatasetMetadata(args.dataset)  # meta/ だけ取得する（動画やデータ本体は落とさない）

    cfg = PI05Config(
        pretrained_path=args.base,
        n_action_steps=args.n_action_steps,
        dtype=torch.bfloat16,
        device="cuda",
        compile_model=False,
        push_to_hub=False,
    )
    # 画像の形状はデータセットでは HWC なので、lerobot の変換（CHW にする）を通してから並べ替える
    feats = dataset_to_policy_features(ds_meta.features)
    cfg.input_features = {
        **{f"observation.images.{cam}": feats[f"observation.images.{cam}"] for cam in CAMERAS},
        "observation.state": feats["observation.state"],
    }
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)

    # lerobot_train.py の processor 生成と同じ: pi05_base の processor を読み、特徴量と統計量を差し替える
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.base,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
                "stats": ds_meta.stats,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
                "stats": ds_meta.stats,
            },
        },
    )

    out = Path(args.out)
    policy.save_pretrained(out)
    preprocessor.save_pretrained(out)
    postprocessor.save_pretrained(out)

    c = policy.config
    print("input :", {k: tuple(v.shape) for k, v in c.input_features.items()})
    print("output:", {k: tuple(v.shape) for k, v in c.output_features.items()})
    print("norm  :", c.normalization_mapping, "| n_action_steps:", c.n_action_steps, "| dtype:", c.dtype)
    print("saved :", out, sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
