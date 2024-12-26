from ultralytics import YOLO
import numpy as np
import logging

# Category mappings
categories = {
    0: "person", 1: "cyclist", 2: "car", 3: "truck", 4: "tram",
    5: "tricycle", 6: "bus", 7: "bicycle", 8: "moped", 9: "motorcycle",
    10: "stroller", 11: "wheelchair", 12: "cart", 13: "trailer",
    14: "construction_vehicle", 15: "recreational_vehicle", 16: "dog",
    17: "barrier", 18: "bollard", 19: "warning_sign", 20: "sentry_box",
    21: "traffic_box", 22: "traffic_cone", 23: "traffic_island",
    24: "traffic_light", 25: "traffic_sign", 26: "debris", 27: "suitcase",
    28: "dustbin", 29: "concrete_block", 30: "machinery", 31: "chair",
    32: "phone_booth", 33: "basket", 34: "misc"
}

def create_gaussian_weight_mask(height, width, sigma=0.5):
    """Create a 2D Gaussian weight mask that emphasizes the center"""
    y = np.linspace(0, height-1, height)
    x = np.linspace(0, width-1, width)
    x, y = np.meshgrid(x, y)

    x0 = width // 2
    y0 = height // 2

    weights = np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))
    return weights / weights.sum()

def get_weighted_depth(depth_map, bbox):
    """Calculate weighted average depth with center emphasis"""
    x1, y1, x2, y2 = map(int, bbox)
    region_depth = depth_map[y1:y2, x1:x2]

    height, width = region_depth.shape
    weight_mask = create_gaussian_weight_mask(height, width, sigma=min(height, width)/4)

    weighted_depth = np.sum(region_depth * weight_mask)
    return weighted_depth

def generate_3d_spatial_graph(image, yolo_model, depth_pipe):
    # Load and process image
    # image = cv2.imread(image_path)
    # pil_image = Image.open(image_path)
    # image = image[0]
    # Get depth map using transformer pipeline
    depth_map = depth_pipe(image)["depth"]
    depth_map = np.array(depth_map)
    depth_map = depth_map.max() - depth_map  # Invert the values

    # Run object detection with custom model
    result = yolo_model(image, verbose=False)[0]

    # Create regions
    image = np.array(image)
    height, width = image.shape[:2]
    x_bounds = [0, width/3, 2*width/3, width]

    # Get depth boundaries
    depth_values = depth_map.flatten()
    d_min, d_max = np.percentile(depth_values, [5, 95])
    total_depth = d_max - d_min
    d_bounds = [d_min, d_min + total_depth/3, d_min + total_depth/1.25, d_max]

    def get_3d_region(bbox, depth):
        center_x = (bbox[0] + bbox[2]) / 2

        # X position
        if center_x < x_bounds[1]:
            x_pos = "left"
        elif center_x < x_bounds[2]:
            x_pos = "middle"
        else:
            x_pos = "right"

        # Depth position
        if depth <= d_bounds[1]:
            d_pos = "front"
        elif depth <= d_bounds[2]:
            d_pos = "middle"
        else:
            d_pos = "back"

        return f"{d_pos} {x_pos}"

    # Map objects to 3D regions
    spatial_relations = {}

    boxes = result.boxes
    for idx, cls in enumerate(boxes.cls):
        bbox = boxes.xyxy[idx].cpu().numpy()
        class_id = int(cls.item())
        class_name = categories[class_id]

        weighted_depth = get_weighted_depth(depth_map, bbox)
        region = get_3d_region(bbox, weighted_depth)

        # Add to spatial relations
        if region not in spatial_relations:
            spatial_relations[region] = []
        spatial_relations[region].append(class_name)

    # Format output
    output_list = []
    for region, objects in sorted(spatial_relations.items()):
        # Count objects by type
        object_counts = {}
        for obj in objects:
            if obj not in object_counts:
                object_counts[obj] = 0
            object_counts[obj] += 1

        # Format the counts into strings
        count_strs = []
        for obj_type, count in object_counts.items():
            if count == 1:
                count_strs.append(f"1 {obj_type}")
            else:
                count_strs.append(f"{count} {obj_type}s")

        # Add formatted region to output
        output_list.append(f"[{region}: {', '.join(count_strs)}]")

    return " ".join(output_list)