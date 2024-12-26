import os
import json

with open('suggestion.json', 'r') as f:
    dataset = json.load(f)

new_suggestion = {}
for key, item in dataset.items():
    if key.startswith("Test_suggestion_"):
        new_suggestion[key] = item

with open('suggestion_filtered.json', 'w') as f:
    json.dump(new_suggestion, f, indent=4)
        