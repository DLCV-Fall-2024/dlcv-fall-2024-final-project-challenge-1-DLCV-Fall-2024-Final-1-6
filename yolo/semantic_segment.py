# import cv2
# import os
# from PIL import Image
# from ultralytics import YOLO
# from datasets import load_dataset

# # 載入 YOLO 模型
# model = YOLO("yolo11l-seg.pt")

# # 載入資料集
# dataset = load_dataset("../data/dlcv_2024_final1", split="train")

# # 定義輸出路徑
# output_dir = "./output_images/train/"
# os.makedirs(output_dir, exist_ok=True)

# for i, data in enumerate(dataset):

#     print(f"Generate segmentation of the {i}th training image. :)")

#     # 取得原始圖片
#     raw_image = data['image']

#     # 執行 YOLO 模型預測
#     results = model.predict(source=raw_image, save=False, save_txt=True, show_boxes=False)  # save=False 表示不儲存到默認路徑

#     # 取得繪製後的圖片
#     result_image = results[0].plot(boxes=False)  # YOLO 結果支援直接生成繪製後的影像 (NumPy array)

#     # 轉換 BGR 到 RGB
#     rgb_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

#     # 儲存為 i_image.jpg
#     save_path = os.path.join(output_dir, f"train_segment_{i}.jpg")
#     Image.fromarray(rgb_image).save(save_path)

#     print(f"Saved segmented image as {save_path}.")

# ---------------------- Validation ----------------------

import cv2
import os
from PIL import Image
from ultralytics import YOLO
from datasets import load_dataset

# 載入 YOLO 模型
model = YOLO("yolo11l-seg.pt")

# 載入資料集
dataset = load_dataset("../data/dlcv_2024_final1", split="validation")

# 定義輸出路徑
output_dir = "./output_images/val/"
os.makedirs(output_dir, exist_ok=True)

for i, data in enumerate(dataset):

    print(f"Generate segmentation of the {i}th validation image. :)")

    # 取得原始圖片
    raw_image = data['image']

    # 執行 YOLO 模型預測
    results = model.predict(source=raw_image, save=False, save_txt=True, show_boxes=False)  # save=False 表示不儲存到默認路徑

    # 取得繪製後的圖片
    result_image = results[0].plot(boxes=False)  # YOLO 結果支援直接生成繪製後的影像 (NumPy array)

    # 轉換 BGR 到 RGB
    rgb_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # 儲存為 i_image.jpg
    save_path = os.path.join(output_dir, f"val_segment_{i}.jpg")
    Image.fromarray(rgb_image).save(save_path)

    print(f"Saved segmented image as {save_path}.")



