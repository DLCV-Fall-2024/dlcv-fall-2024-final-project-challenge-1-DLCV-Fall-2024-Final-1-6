import os
import json
import numpy as np

with open('submission.json', 'r') as f:
    dataset = json.load(f)

assert len(dataset)==900
should_exist = np.full((3, 300), 1)

for key, value in dataset.items():
    idx = int(key.split('_')[-1])
    task_type = key.split('_')[1]
    task_idx = 0 if task_type == "general" else 1 if task_type == "regional" else 2

    assert should_exist[task_idx, idx] == 1
    should_exist[task_idx, idx] == 0

assert np.any(should_exist == 1)