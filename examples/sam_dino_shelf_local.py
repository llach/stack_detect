import os
import time
import cv2
import numpy as np
from PIL import Image

# Imports from your stack project
from stack_detect.helpers.sam2_model import SAM2Model
from stack_detect.helpers.dino_model import DINOModel, plot_boxes_to_image

# ---------- PATH SETUP ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "./gowns_shelf.png")

ckp_root = os.path.expanduser(f"{os.environ['HOME']}/repos/ckp")

DINO_PREFIX = os.path.join(ckp_root, "")
SAM_CKPT = os.path.join(ckp_root, "sam2.1_hiera_large.pt")
SAM_CFG = os.path.join(ckp_root, "sam2.1_hiera_l.yaml")

# ---------- INIT MODELS ----------
dino = DINOModel(prefix=DINO_PREFIX, cpu_only=False)
sam = SAM2Model()#checkpoint=SAM_CKPT, model_cfg=SAM_CFG)

# ---------- LOAD IMAGE ----------
image_pil = Image.open(image_path).convert("RGB")
img_raw = np.array(image_pil)

# ---------- RUN DINO ----------
dino_start = time.time()
print("Running DINO ...")
boxes_px, pred_phrases, confidences = dino.predict(image_pil, "detect all stacks of clothing")
print(f"DINO took {round(time.time() - dino_start, 2)}s")

# Draw boxes
image_with_box = plot_boxes_to_image(image_pil.copy(), boxes_px, pred_phrases)[0]
cv2.imshow("DINO detections", cv2.cvtColor(np.array(image_with_box), cv2.COLOR_RGB2BGR))

# ---------- SELECT BOX ----------
box_idx = 0
box = boxes_px[box_idx]

# ---------- RUN SAM ----------
sam_start = time.time()
print("Running SAM ...")
masks = sam.predict(img_raw)
print(f"SAM took {round(time.time() - sam_start, 2)}s")

# ---------- COMBINE RESULTS ----------
img_overlay, _, line_center = SAM2Model.detect_stack(img_raw, masks, box)

# ---------- SHOW / SAVE ----------
cv2.imshow("DINO + SAM Combined", cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))

# Optional: save output to file
out_path = os.path.join(script_dir, "dino_sam_combined.png")
cv2.imwrite(out_path, cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR))
print(f"Combined image saved to {out_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()
