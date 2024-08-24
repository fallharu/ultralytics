import torch
import numpy as np
import cv2
from ultralytics import YOLO
import pandas as pd
import sys
import os

# プロジェクトのベースディレクトリを取得
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# モデルのロード
model = YOLO('yolov8n.pt')
model.to("cuda")

# 入力ビデオのパス
video_path = sys.argv[1]

frame = cv2.imread(video_path)
# 予測の実行
results = model.track(video_path, save=True, conf=0.05, classes=2, save_txt=True, hide_conf=True, show=True)

# すべてのトラッキングIDとxyxyデータを格納するリストを初期化
all_frame_now = []
all_frame_box = []
all_track_ids = []
all_xyxy_data = []
all_class_ids = []

for i, result in enumerate(results):
    if result.boxes.id is not None:
        xyxy_data = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        # トラッキングIDとxyxyデータをそれぞれのリストに追加
        all_frame_now.extend([i+1]*len(track_ids))
        all_frame_box.extend([i+1]*len(track_ids))  # frame_box_counterをi+1に変更
        all_track_ids.extend(track_ids)
        all_xyxy_data.extend(xyxy_data)
        all_class_ids.extend(class_ids)
    else:
        # 物体が検出されなかったフレームに対しては、NaNを追加
        all_frame_now.append(i+1)
        all_frame_box.append(np.nan)
        all_track_ids.append(np.nan)
        all_xyxy_data.append([np.nan, np.nan, np.nan, np.nan])
        all_class_ids.append(np.nan)

# 保存先のパス
save_dir_detailed = os.path.join(BASE_DIR, "..", "Temporary_Data", "csv", "Detailed_data")
save_dir_editing = os.path.join(BASE_DIR, "..", "Temporary_Data", "csv", "Editing")
os.makedirs(save_dir_detailed, exist_ok=True)
os.makedirs(save_dir_editing, exist_ok=True)

save_path_ids = os.path.join(save_dir_detailed, "track_ids.csv")
save_path_xyxy = os.path.join(save_dir_detailed, "xyxy_data.csv")
save_path_frame_now = os.path.join(save_dir_detailed, "frame_now.csv")
save_path_frame_box = os.path.join(save_dir_detailed, "frame_box.csv")
save_path_class_ids = os.path.join(save_dir_detailed, "class_ids.csv")

# すべてのフレーム番号、トラッキングIDとxyxyデータをそれぞれのCSVファイルとして保存
np.savetxt(save_path_frame_now, np.array(all_frame_now), delimiter=",", fmt='%f')
np.savetxt(save_path_ids, np.array(all_track_ids), delimiter=",", fmt='%f')
np.savetxt(save_path_xyxy, np.array(all_xyxy_data), delimiter=",", fmt='%f')
np.savetxt(save_path_frame_box, np.array(all_frame_box), delimiter=",", fmt='%f')
np.savetxt(save_path_class_ids, np.array(all_class_ids), delimiter=",", fmt='%f')

# 保存したフレーム番号、トラッキングIDとxyxyデータのCSVファイルを読み込む
df_frame_now = pd.read_csv(save_path_frame_now, header=None)
df_track_ids = pd.read_csv(save_path_ids, header=None)
df_xyxy_data = pd.read_csv(save_path_xyxy, header=None)
df_frame_box = pd.read_csv(save_path_frame_box, header=None)
df_class_ids = pd.read_csv(save_path_class_ids, header=None)

# 列名を定義
column_names = ['frame_now', 'track_ids', 'x1', 'y1', 'x2', 'y2', 'frame_box', 'class_ids']

# フレーム番号、トラッキングID、xyxyデータ、クラスIDを結合
df_combined = pd.concat([df_frame_now, df_track_ids, df_xyxy_data, df_frame_box, df_class_ids], axis=1)

# 列名をデータフレームに設定
df_combined.columns = column_names

# 結合したデータを新しいCSVファイルに保存（列名を含む）
df_combined.to_csv(os.path.join(save_dir_editing, "combined_data.csv"), index=False)
