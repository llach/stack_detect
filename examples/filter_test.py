import os
import cv2
import pickle
import numpy as np

from stack_detect.helpers.sam2_model import (
    SAM2Model,
    draw_anns, 
    calculate_mask_centers, 
    find_masks_in_box, 
    sort_masks_by_horizontal_position,
    get_border_pixels
)

with open(f"{os.environ['HOME']}/stack.pkl", "rb") as f:
    data = pickle.load(f)
img_raw, boxes_px, masks = data

img_overlay, line_pixels, line_center = SAM2Model.detect_stack(img_raw, masks, boxes_px[0])

cv2.imshow("Line Center", cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)