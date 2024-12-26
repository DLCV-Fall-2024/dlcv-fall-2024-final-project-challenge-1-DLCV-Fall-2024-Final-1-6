import json

def convert_to_conversation_format(input_data_path, output_data_path):
    with open(input_data_path, 'r') as f:
        data = json.load(f)

    conversations = []

    for image_id, objects in data.items():
        human_instruction = (
            "<image>\n"
            "Describe the objects influencing the ego car's behavior based on their appearance, position, "
            "depth category (e.g., close range, mid range, far range), and category name (e.g., car, pedestrian)."
        )
        gpt_response = "In this scene:\n"
        for obj in objects:
            gpt_response += (
                f"- A {obj['depth_category']} {obj['category_name']} located to the {obj['position']} "
                f"with confidence {obj['confidence']:.2f}. The bounding box coordinates are {obj['bbox']}.\n"
            )
        
        conversations.append({
            "image": f"path/to/{image_id}.jpg",
            "conversations": [
                { "from": "human", "value": human_instruction },
                { "from": "gpt", "value": gpt_response }
            ]
        })

    with open(output_data_path, 'w') as f:
        json.dump(conversations, f, indent=2)

convert_to_conversation_format("preprocessed_data.json", "formatted_data.json")
