import os
import cv2
import pickle
import numpy as np

from stack_detect.helpers.sam2_model import (
    draw_anns, 
    calculate_mask_centers, 
    find_masks_in_box, 
    sort_masks_by_horizontal_position,
    get_border_pixels
)

with open(f"{os.environ['HOME']}/stack.pkl", "rb") as f:
    data = pickle.load(f)
img_raw, boxes_px, masks = data

### draw masks
msk_w = 0.4
img_anns = draw_anns(masks)
img_overlay = np.clip((1-msk_w)*img_raw + msk_w*img_anns, 0, 255).astype(np.uint8)

### draw DINO box
x0, y0, x1, y1 = boxes_px[0]
cv2.rectangle(img_overlay, (x0, y0), (x1, y1), (255, 0, 255), 2)

### sort masks based on horizontal center position
sorted_masks = sort_masks_by_horizontal_position(masks, left_is_up=True)

### find masks where EVERY pixel falls inside the DINO box, and all also those outside
masks_inside, masks_outside = find_masks_in_box(sorted_masks, boxes_px[0])

### select the first cluster, get line pixels and a mask for drawing
upper_layer_mask = masks_inside[0]
line_mask, line_pixels = get_border_pixels(upper_layer_mask["segmentation"], left_is_up=True)
line_center = np.mean(line_pixels, axis=0).astype(np.uint64)

# draw line and grasp center
img_overlay[line_mask] = [255,0,255]
cv2.circle(img_overlay, line_center, 3, (100,100,100), -1)

# draw mask centers, the ones inside are also annotated by their index from sorting
centers_inside = calculate_mask_centers(masks_inside)
for i, c in enumerate(centers_inside):
    cv2.putText(img_overlay,f'{i}', c, cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,0),2,cv2.LINE_AA)
    cv2.circle(img_overlay, c, 2, (255,0,0), -1)

centers_outside = calculate_mask_centers(masks_outside)
for c in centers_outside:
    cv2.circle(img_overlay, c, 2, (255,255,255), -1)

cv2.imshow("Line Center", cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)