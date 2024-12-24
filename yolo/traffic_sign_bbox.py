# import cv2
# from PIL import Image
# import torch
# from ultralytics import YOLO

# model = YOLO("yolo11x.pt")

# # Train the model
# model.train(
#     data="data.yaml",  # path to dataset YAML
#     epochs=40,  # number of training epochs
#     imgsz=640,  # training image size
#     device="cuda" if torch.cuda.is_available() else "cpu",  # CUDA training
# )

# results = model.predict(source="./test_dataset", save=True, save_txt=True, show_labels=True)  # Display preds. Accepts all YOLO predict arguments

# for i, result in enumerate(results):
#     result.save(f"./output/traffic_sign_{i}.jpg")

# # ---------------------- Rag - Training ----------------------

# import cv2
# import os
# from PIL import Image
# from ultralytics import YOLO
# from datasets import load_dataset

# # 載入 YOLO 模型
# model = YOLO("traffic_sign.pt")

# # 載入資料集
# dataset = load_dataset("../data/dlcv_2024_final1", split="train")

# # 定義輸出路徑
# output_dir = "./bbox_output_images/train/"
# os.makedirs(output_dir, exist_ok=True)

# for i, data in enumerate(dataset):

#     # if i == 10:
#     #     break

#     print(f"Generate traffic sign bbox of the {i}th train image. :)")

#     # 取得原始圖片
#     raw_image = data['image']

#     # 執行 YOLO 模型預測
#     results = model.predict(source=raw_image, save=False, save_txt=True, show_boxes=False)  # save=False 表示不儲存到默認路徑

#     # 取得繪製後的圖片
#     result_image = results[0].plot(boxes=True, labels=False)  # YOLO 結果支援直接生成繪製後的影像 (NumPy array)

#     # 轉換 BGR 到 RGB
#     rgb_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

#     # 儲存為 i_image.jpg
#     save_path = os.path.join(output_dir, f"train_bbox_{i}.jpg")
#     Image.fromarray(rgb_image).save(save_path)

#     print(f"Saved traffic sign bbox image as {save_path}.")

# ---------------------- Rag - Validation ----------------------

import cv2
import os
from PIL import Image
from ultralytics import YOLO
from datasets import load_dataset

# 載入 YOLO 模型
model = YOLO("traffic_sign.pt")

# 載入資料集
dataset = load_dataset("../data/dlcv_2024_final1", split="validation")

# 定義輸出路徑
output_dir = "./bbox_output_images/val/"
os.makedirs(output_dir, exist_ok=True)

for i, data in enumerate(dataset):

    # if i == 10:
    #     break

    print(f"Generate traffic sign bbox of the {i}th validation image. :)")

    # 取得原始圖片
    raw_image = data['image']

    # 執行 YOLO 模型預測
    results = model.predict(source=raw_image, save=False, save_txt=True, show_boxes=False)  # save=False 表示不儲存到默認路徑

    # 取得繪製後的圖片
    result_image = results[0].plot(boxes=True, labels=False)  # YOLO 結果支援直接生成繪製後的影像 (NumPy array)

    # 轉換 BGR 到 RGB
    rgb_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # 儲存為 i_image.jpg
    save_path = os.path.join(output_dir, f"val_bbox_{i}.jpg")
    Image.fromarray(rgb_image).save(save_path)

    print(f"Saved traffic sign bbox image as {save_path}.")