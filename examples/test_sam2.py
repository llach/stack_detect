import os
import time
import pathlib
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

image = iio.imread(pathlib.Path(__file__).parent.joinpath("stack_full.png"))

checkpoint = f"{os.environ['HOME']}/repos/ckp/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

mask_generator_2 = SAM2AutomaticMaskGenerator(
    model=build_sam2(model_cfg, checkpoint),
    points_per_side=24,
    points_per_batch=44,
    # pred_iou_thresh=0.7,
    stability_score_thresh=0.9,
    # stability_score_offset=0.7,
    # crop_n_layers=2,
    # box_nms_thresh=0.7,
    crop_n_points_downscale_factor=4,
    min_mask_region_area=25.0,
    # use_m2m=True,
)

start = time.time()
masks2 = mask_generator_2.generate(image)
print(f"took {time.time()-start}")

plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks2)
plt.axis('off')
plt.show() 