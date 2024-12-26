import os
import json
from tqdm import trange
import numpy as np
import subprocess

AVAILABLE_DEVICES = [0, 1, 2, 3]
DATASET_PATH = '/home/twszmxa453/CVPR2025/dlcv_json_files/test.json'
OUTPUT_DIR = '/home/twszmxa453/CVPR2025/dlcv_splitted_test_jsons'
SUBMISSION_FOLDER = "/home/twszmxa453/CVPR2025/submissions"
FINAL_SUBMISSION_PATH = "/home/twszmxa453/CVPR2025/submissions/submission.json"


def find_object_with_id(data, target_id):
    # print("target_id", target_id)
    result = next((item for item in data if item['id'] == target_id), None)
    return result

# Step 1: Split the dataset
def split_dataset():
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    device_length = len(AVAILABLE_DEVICES)

    cut_points = np.linspace(0, 300, device_length + 1, dtype=np.int16)
    print("Cut points:", cut_points)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    splitted_dataset = []
    cur_device_idx = 0
    for i in range(300):
        object_1 = find_object_with_id(dataset, f"Test_general_{i}")
        object_2 = find_object_with_id(dataset, f"Test_regional_{i}")
        object_3 = find_object_with_id(dataset, f"Test_suggestion_{i}")

        splitted_dataset.append(object_1)
        splitted_dataset.append(object_2)
        splitted_dataset.append(object_3)

        if i == cut_points[cur_device_idx+1]-1:
            split_path = os.path.join(OUTPUT_DIR, f"data_{AVAILABLE_DEVICES[cur_device_idx]}.json")
            print(f"Saving split dataset for device {AVAILABLE_DEVICES[cur_device_idx]} to {split_path} with size {len(splitted_dataset)}")
            with open(split_path, "w") as f:
                json.dump(splitted_dataset, f, indent=4)
            cur_device_idx += 1
            splitted_dataset = []
        
def merge_submissions():
    merged_submissions = {}
    for device in AVAILABLE_DEVICES:
        output_path = os.path.join(SUBMISSION_FOLDER, f"submission_{device}.json")
        with open(output_path, "r") as f:
            submission = json.load(f)
        merged_submissions.update(submission)
    with open(FINAL_SUBMISSION_PATH, "w") as f:
        json.dump(merged_submissions, f, indent=4)


def main():
    # Step 1: Split the dataset
    split_dataset()

    # Step 2: Parallelize inference
    processes = []
    for device in AVAILABLE_DEVICES:
        split_name = f"data_{device}.json"
        split_path = os.path.join(OUTPUT_DIR, split_name)
        output_path = os.path.join(SUBMISSION_FOLDER, f"submission_{device}.json")
        # Construct the command for inference
        cmd = [
            "bash", "parallel_inference.sh",
            split_path,
            output_path
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device)  # Set the GPU for this process
        print(f"Launching process for device {device} with dataset {split_name} on GPU {device}")
        # Start the process
        proc = subprocess.Popen(cmd)
        processes.append(proc)

    # Wait for all processes to complete
    for proc in processes:
        proc.wait()

    print("All inference processes completed.")

    # Step 3: Merge all the JSONs
    merge_submissions()

if __name__ == "__main__":
    main()