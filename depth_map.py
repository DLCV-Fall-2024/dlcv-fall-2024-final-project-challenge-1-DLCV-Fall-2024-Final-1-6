# from transformers import DPTImageProcessor, DPTForDepthEstimation
# from PIL import Image
# import torch
# import cv2
# import numpy as np

# processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
# model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# # 加載圖片
# image = Image.open("dataset/test/images/Test_general_0.png").convert("RGB")
# inputs = processor(images=image, return_tensors="pt")


# # 深度預測
# with torch.no_grad():
#     outputs = model(**inputs)
# depth = outputs.predicted_depth
# prediction = torch.nn.functional.interpolate(
#                     depth.unsqueeze(1),
#                     size=image.size[::-1],
#                     mode="bicubic",
#                     align_corners=False,
#               ).squeeze()
# output = prediction.cpu().numpy()
# formatted = (output - np.min(output)) / (np.max(output) - np.min(output))
# formatted = (formatted * 255).astype('uint8')
# cv2.imwrite("Test_general_0_depth.png", formatted)


from transformers import AutoImageProcessor, DPTForDepthEstimation
from PIL import Image
import torch
import numpy as np
import cv2

image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-small-kitti")
model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-small-kitti")

# image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
# model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")


image = Image.open("dataset/test/images/Test_general_0.png").convert("RGB")
inputs = image_processor(images=image, return_tensors="pt")


# 深度預測
with torch.no_grad():
    outputs = model(**inputs)
    depth = outputs.predicted_depth
prediction = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
              ).squeeze()
output = prediction.cpu().numpy()
formatted = (output - np.min(output)) / (np.max(output) - np.min(output))
formatted = (formatted * 255).astype('uint8')


save_path = "Test_general_0_depth.png"
cv2.imwrite(save_path, formatted)

# midas
# import torch
# import cv2
# import numpy as np
# from torchvision.transforms import Compose, Normalize, ToTensor

# model_type = "DPT_Large"
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform

# image_path = "dataset/test/images/Test_general_0.png"
# image = cv2.imread(image_path)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# input_batch = transform(image_rgb).to(device)

# with torch.no_grad():
#     prediction = midas(input_batch)
#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=image_rgb.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()

# depth_map = prediction.cpu().numpy()
# normalized_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
# depth_gray = (normalized_depth * 255).astype("uint8")

# output_path = "Test_general_0_depth_midas.png"
# cv2.imwrite(output_path, depth_gray)
