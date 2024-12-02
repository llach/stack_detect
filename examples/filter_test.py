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
    get_border_pixels,
    filter_masks_by_size
)

with open(f"{os.environ['HOME']}/repos/bags/line_broken.pkl", "rb") as f:
    data = pickle.load(f)
img_raw, boxes_px, masks = data


# sort masks based on horizontal center position
# sorted_masks = sort_masks_by_horizontal_position(masks, left_is_up=True)

# ### find masks where EVERY pixel falls inside the DINO box, and all also those outside
# masks_inside, masks_outside = find_masks_in_box(sorted_masks, boxes_px[0])

# ### we discard masks that are not wide enough (small features in the background) and too tall (sometimes the stack itself is detected as a whole)
# masks_inside_ok, masks_inside_not_ok = filter_masks_by_size(masks_inside, boxes_px[0], rotated=True)

# mask_w = 0.6
# for i, m in enumerate(masks_inside_ok):
#     img_anns = draw_anns([m])
#     img_overlay = np.clip((1-mask_w)*img_raw + mask_w*img_anns, 0, 255).astype(np.uint8)

#     cv2.imshow(f"Mask {i}", cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
#     cv2.waitKey(0)
#     cv2.destroyWindow(f"Mask {i}")

img_overlay, line_pixels, line_center = SAM2Model.detect_stack(img_raw, masks, boxes_px[0])

cv2.imshow("Line Center", cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)