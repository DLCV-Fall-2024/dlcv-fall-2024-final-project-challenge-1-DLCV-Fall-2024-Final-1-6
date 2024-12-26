import numpy as np
import random

import inflect

# Create an inflect engine
p = inflect.engine()

# Function to handle singular and plural nouns
def make_plural(noun):
    # Check if the noun is already plural
    if not p.singular_noun(noun):
        # If it is singular, convert to plural
        return p.plural(noun)
    return noun

CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "van",
    "SUV",
    "trailer",
    "moped",
    "ambulance",
    "construction vehicle",
    "pedestrian",
    "cyclist",
    "motorcyclists",
    "road users",
    "red traffic light",
    "traffic light",
    "parking sign",
    "warning traffic sign",
    "directional traffic sign",
    "traffic box",
    "sentry box",
    "traffic cone",
    "traffic island",
    "barrier",
    "bollard",
    "debris",
    "machinery",
    "dustbin",
    "concrete block",
    "dog",
    "chair",
    "phone booth",
    "streetlights"
]

def save_qa(q, a):
    return [
        {"from": "human", "value": q},
        {"from": "gpt", "value": a},
    ]


def get_pretrain_gt_conversations(image_name, image_data, category_count, near_category_count):
    number_of_objects = len(image_data)
    conversations = []

    if number_of_objects < 2:
        print(f"No objects detected in image {image_name}.")
        return conversations
    
    category_with_objetcs = [category for category, count in category_count.items() if count > 0]
    category_with_near_objects = [category for category, count in near_category_count.items() if count > 0]

    # QA 1
    q = "<image>\nPlease describe the number of objects in each category within the image based on the provided bounding boxes information."
    a = "There are"
    for index, category in enumerate(category_with_objetcs):
        if category_count[category] == 1:
            category_noun = p.singular_noun(make_plural(category))
        else:
            category_noun = make_plural(category)
            
        if index == len(category_with_objetcs) - 1:
            a += f" and {category_count[category]} {category_noun} in front of the ego car."
        else:
            a += f" {category_count[category]} {category_noun},"

    conversations.append(save_qa(q,a))

    # QA 2 ~ 4
    # random select two objects
    positions = ["left", "middle", "right"]
    random_integers = random.choices(list(range(len(category_with_objetcs))), k=3)
    for k, position in enumerate(positions):
        count_ans = 0
        if position == "left":
            target_category = category_with_objetcs[random_integers[k]]
            q = f"<image>\nHow many {make_plural(target_category)} are positioned ahead and to the left of the ego car?"
            for data in image_data:
                if data["category_name"] == target_category and data["position"] == "left":
                    count_ans += 1
                    
            if count_ans == 0:
                a = f"No {make_plural(category_with_objetcs[random_integers[k]])} are positioned ahead and to the left of the ego car."
            elif count_ans == 1:
                a = f"Only one {p.singular_noun(make_plural(category_with_objetcs[random_integers[k]]))} is positioned ahead and to the left of the ego car."
            else:
                a = f"There are {count_ans} {make_plural(category_with_objetcs[random_integers[k]])} positioned ahead and to the left of the ego car."
            
        if position == "middle":
            target_category = category_with_objetcs[random_integers[k]]
            q = f"<image>\nHow many {make_plural(category_with_objetcs[random_integers[k]])} are positioned straight in front of the ego car?"
            for data in image_data:
                if data["category_name"] == target_category and data["position"] == "middle":
                    count_ans += 1
                    
            if count_ans == 0:
                a = f"No {make_plural(category_with_objetcs[random_integers[k]])} are positioned straight in front of the ego car."
            elif count_ans == 1:
                a = f"Only one {p.singular_noun(make_plural(category_with_objetcs[random_integers[k]]))} is positioned straight in front of the ego car."
            else:
                a = f"There are {count_ans} {make_plural(category_with_objetcs[random_integers[k]])} positioned straight in front of the ego car."
            
        if position == "right":
            target_category = category_with_objetcs[random_integers[k]]
            q = f"<image>\nHow many {make_plural(category_with_objetcs[random_integers[k]])} are positioned ahead and to the right of the ego car?"
            for data in image_data:
                if data["category_name"] == target_category and data["position"] == "right":
                    count_ans += 1
                    
            if count_ans == 0:
                a = f"No {make_plural(category_with_objetcs[random_integers[k]])} are positioned ahead and to the right of the ego car."
            elif count_ans == 1:
                a = f"Only one {p.singular_noun(make_plural(category_with_objetcs[random_integers[k]]))} is positioned ahead and to the right of the ego car."
            else:
                a = f"There are {count_ans} {make_plural(category_with_objetcs[random_integers[k]])} positioned ahead and to the right of the ego car."
        
        conversations.append(save_qa(q,a))
        
        

    # QA 5 ~ 7
    nearest_category = dict()
    max_depth = dict()
    max_depth["left"] = 0
    max_depth["middle"] = 0
    max_depth["right"] = 0
    for data in image_data:
        object_position = data["position"]
        if data["depth_value"] > max_depth[object_position]:
            nearest_category[object_position] = data["category_name"]
            max_depth[object_position] = data["depth_value"]
    
    
    q = "<image>\nWhat is the nearest object to the left of the ego car?"
    if max_depth["left"] == 0:
        a = "There is nothing to the left of the ego car."
    else:
        ans_object = p.singular_noun(make_plural(nearest_category["left"]))
        a = f"The nearest object to the left of the ego car is {ans_object}."
    conversations.append(save_qa(q,a))
    
    q = "<image>\nWhat is the nearest object directly in front of the ego car?"
    if max_depth["middle"] == 0:
        a = "There is nothing directly in front of the ego car."
    else:
        ans_object = p.singular_noun(make_plural(nearest_category["middle"]))
        a = f"The nearest object directly in front of the ego car is {ans_object}."
    conversations.append(save_qa(q,a))
        
    q = "<image>\nWhat is the nearest object to the right of the ego car?"
    if max_depth["right"] == 0:
        a = "There is nothing to the right of the ego car."
    else:
        ans_object = p.singular_noun(make_plural(nearest_category["right"]))
        a = f"The nearest object to the right of the ego car is {ans_object}."
    conversations.append(save_qa(q,a))
    
        
    return conversations
