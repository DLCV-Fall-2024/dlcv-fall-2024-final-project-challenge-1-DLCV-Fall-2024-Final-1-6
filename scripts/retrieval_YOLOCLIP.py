import torch
from ultralytics import YOLO
from datasets import load_dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
import json
from scipy.optimize import linear_sum_assignment

class ImageRetrieval:
    def __init__(self, yolo_model="yolo11x.pt", clip_model="openai/clip-vit-base-patch32", max_objects=5, golden_json="./stage2/golden.json", mapping_json="./stage2/train_idmap.json"):
        # Initialize models
        self.yolo = YOLO(yolo_model)
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model)
        self.processor = CLIPProcessor.from_pretrained(clip_model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        self.max_objects = max_objects

        with open(golden_json, "r") as f:
            self.golden = json.load(f)
        with open(mapping_json, "r") as f:
            self.mapping = json.load(f)

        self.rev_mapping = {}

        # Iterate through the original dictionary
        for key, value in self.mapping.items():
            # If the value is not in the new dict, create a new list
            if value not in self.rev_mapping:
                self.rev_mapping[value] = []
            # Append the original key to the list
            self.rev_mapping[value].append(key)

        # Initialize feature banks
        self.feature_banks = {
            'general': [],
            'regional': [],
            'suggestion': []
        }
        self.image_ids = {
            'general': [],
            'regional': [],
            'suggestion': []
        }
        self.image_captions = {
            'general': [],
            'regional': [],
            'suggestion': []
        }
        self.image_lookup = {}

        # Build indices
        self.build_all_indices()

    def extract_features(self, image):
        features = []

        # 1. Extract features from the full image
        input_full = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            full_features = self.clip_model(pixel_values=input_full.pixel_values).image_embeds
            full_features = full_features/full_features.norm(p=2, dim=-1, keepdim=True)
            features.append(full_features.cpu().numpy().flatten())

        # 2. Use YOLO to detect objects
        results = self.yolo(image, verbose=False)
        object_features = []
        if len(results) > 0:
            # Get top max_objects objects with highest confidence
            boxes = results[0].boxes
            conf = boxes.conf.cpu().numpy()
            indices = np.argsort(conf)[-self.max_objects:]
            boxes = boxes.xyxy.cpu().numpy()[indices]

            for box in boxes:
                cropped = image.crop(box)
                input_obj = self.processor(images=cropped, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    obj_features = self.clip_model(pixel_values=input_obj.pixel_values).image_embeds
                    obj_features = obj_features/obj_features.norm(p=2, dim=-1, keepdim=True)
                    object_features.append(obj_features.cpu().numpy().flatten())

        # Pad to fixed length
        feature_dim = full_features.shape[1]
        while len(object_features) < self.max_objects:
            object_features.append(np.zeros(feature_dim))

        # Ensure only taking max_objects objects
        object_features = object_features[:self.max_objects]

        # Concatenate all features
        all_features = np.concatenate([features[0]] + object_features)
        return all_features

    def build_all_indices(self, dataset_path="ntudlcv/dlcv_2024_final1", split="train", sample_size=2000):
        #Train_general_4883
        dataset = load_dataset(dataset_path, split=split)
        indices = range(sample_size)
        sampled_dataset = [dataset[int(i)] for i in indices]

        # print(f"Building image lookup dictionary for {sample_size} images...")
        # self.image_lookup = {item['id']: item for item in sampled_dataset}

        for category in ['general', 'regional', 'suggestion']:
            self.feature_banks[category] = []
            self.image_ids[category] = []
            self.image_captions[category] = []

        for item in tqdm(sampled_dataset, desc="Building indices for all categories"):
            image_id = item['id']
            image = item['image']

            if 'Train_general' in image_id:
                category = 'general'
            # elif 'Train_regional' in image_id:
            #     category = 'regional'
            # elif 'Train_suggestion' in image_id:
            #     category = 'suggestion'
            else:
                continue

            features = self.extract_features(image)

            if len(features) > 0:
                self.feature_banks[category].append(features)
                self.image_ids[category].append(image_id)
                self.image_captions[category].append(item['conversations'][1]['value'])

        for category in self.feature_banks:
            if self.feature_banks[category]:
                self.feature_banks[category] = np.array(self.feature_banks[category])

    def retrieve(self, query_id, query_image, k=5, category=None):
        query_features = self.extract_features(query_image)

        if len(query_features) == 0:
            return []

        results = self._retrieve_from_category(query_features, "general", k)

        select = 0  # 默認使用第一個結果
        if query_id in self.golden.keys():
            try:
                if query_id.replace("suggestion", "general") == results[0]:
                    select = 1  # Train data, general or suggestion
                elif "regional" in query_id and query_id in self.rev_mapping.get(results[0], []):
                    select = 1  # Train data, regional
            except:
                select = 0  # 如果有任何錯誤，使用第一個結果

        result_id = results[select]

        # 同樣處理 caption 的部分
        try:
            if "suggestion" in query_id:
                caption = self.golden[result_id.replace("general", "suggestion")]
            elif "regional" in query_id:
                caption = self.golden.get(self.rev_mapping.get(result_id, [None])[0], self.golden[result_id])
            else:
                caption = self.golden[result_id]
        except:
            caption = self.golden[result_id]  # 如果出錯就直接使用 result_id 對應的 caption

        return result_id, caption

    def _retrieve_from_category(self, query_features, category, k):
        if self.feature_banks[category] is None or len(self.feature_banks[category]) == 0:
            return []

        # Separate global features and object features
        feature_dim = len(query_features) // (self.max_objects + 1)
        query_global = query_features[:feature_dim]
        query_objects = query_features[feature_dim:].reshape(self.max_objects, feature_dim)

        similarities = []
        for bank_features in self.feature_banks[category]:
            # Separate global features and object features from database
            bank_global = bank_features[:feature_dim]
            bank_objects = bank_features[feature_dim:].reshape(self.max_objects, feature_dim)

            # Calculate similarity of global features
            global_sim = np.dot(query_global, bank_global) / (
                np.linalg.norm(query_global) * np.linalg.norm(bank_global)
            )

            # Calculate similarity matrix between objects
            object_sim_matrix = np.zeros((self.max_objects, self.max_objects))
            for i in range(self.max_objects):
                for j in range(self.max_objects):
                    # Only calculate if both object features are non-zero vectors
                    if not np.all(query_objects[i] == 0) and not np.all(bank_objects[j] == 0):
                        object_sim_matrix[i,j] = np.dot(query_objects[i], bank_objects[j]) / (
                            np.linalg.norm(query_objects[i]) * np.linalg.norm(bank_objects[j])
                        )

            # Use Hungarian algorithm to find optimal object matching
            row_ind, col_ind = linear_sum_assignment(-object_sim_matrix)  # Negative sign because algorithm minimizes

            # Calculate average similarity of object matches
            object_sim = object_sim_matrix[row_ind, col_ind].mean()

            # Combine global and object similarities
            final_sim = 0.5 * global_sim + 0.5 * object_sim
            similarities.append(final_sim)

        if not similarities:
            return []

        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.image_ids[category][i] for i in top_k_indices]
def main():
    # 初始化檢索系統
    retriever = ImageRetrieval(max_objects=5)
    
    # 載入測試資料
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="test")
    test_index = 5
    test_image = dataset[test_index]['image']
    test_id = dataset[test_index]['id']
    print('test: ', test_id)
    
    # 創建輸出目錄
    os.makedirs(str(test_id), exist_ok=True)
    test_image.save(os.path.join(str(test_id), 'test_image.jpg'))
    
    # 檢索相似圖片
    general_results = retriever.retrieve(test_image, k=5, category="general")
    print("Similar general image IDs:", general_results)
    
    # 儲存結果
    for i, img_id in enumerate(general_results):
        print(i, img_id)
        if img_id in retriever.image_lookup:
            result_img = retriever.image_lookup[img_id]
            result_img.save(os.path.join(str(test_id), f'general_result_{i}.jpg'))

if __name__ == "__main__":
    main()