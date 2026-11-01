"""RoboCasa GetToastedBread の実験で共有する定数。"""

# 50 タスク統合の人間デモ（target split）。GetToastedBread は 506 本（episode index 13173〜13678）
HUB_REPO = "lerobot/robocasa_target_human_unified"
TASK = "GetToastedBread"
# lerobot-eval で環境が渡す指示文（robocasa get_toasted_bread.py の get_ep_meta()["lang"] と一致することを確認済み）
INSTRUCTION = "Start the toaster. Once the lever pops up, take the bread to the plate on the dining counter."
# pi05_base のカメラの並び base / left_wrist / right_wrist に合わせた順（データセットの並びは left, right, wrist）
CAMERAS = ["robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"]
# diagnose_toast_eval.py が記録する段階（到達順）
STAGES = ["lever_down", "turned_on", "toaster_on", "popped", "grasped", "lifted", "in_plate", "success"]
