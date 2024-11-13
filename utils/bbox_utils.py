def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    # print(f"Center: ({center_x}, {center_y})")  # Debugging
    return center_x, center_y

def get_bbox_width(bbox):
    width = bbox[2] - bbox[0]
    # print(f"BBox Width: {width}")  # Debugging
    return width
