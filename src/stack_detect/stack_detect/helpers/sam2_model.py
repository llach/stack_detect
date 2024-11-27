import os
import numpy as np
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def calculate_mask_centers(mask_dicts):
    """
    Calculate the pixel-space centers of masks in a list of dictionaries.

    Args:
        mask_dicts (list): A list of dictionaries, each with a key "segmentation"
                           containing a 2D boolean numpy array.

    Returns:
        list: A list of tuples representing the pixel-space centers (y, x) for each mask.
    """
    centers = []
    for mask_dict in mask_dicts:
        segmentation = mask_dict["segmentation"]
        if not isinstance(segmentation, np.ndarray) or segmentation.dtype != bool:
            raise ValueError("The 'segmentation' key must contain a 2D boolean numpy array.")
        
        # Get the coordinates of the mask pixels
        rows, cols = np.where(segmentation)
        
        # Calculate the center of mass
        if rows.size > 0 and cols.size > 0:
            center_y = int(np.mean(rows))
            center_x = int(np.mean(cols))
            centers.append((center_x, center_y))
        else:
            # Handle empty masks
            centers.append((None, None))
    
    return centers

def draw_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3)*[0,255,255] # less red so if we draw over masks, we have higher contrast
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    return img.astype(np.uint8)

def find_masks_in_box(mask_dicts, box):
    """
    Find masks that are fully contained within a bounding box and masks that are not.

    Args:
        mask_dicts (list of dict): List of dictionaries, each with a key "segmentation"
                                   containing a 2D boolean array representing the mask.
        box (tuple): Bounding box in pixel space (x0, y0, x1, y1).

    Returns:
        tuple: (contained_masks, not_contained_masks)
               - contained_masks: List of dicts for masks fully contained in the box.
               - not_contained_masks: List of dicts for masks not fully contained in the box.
    """
    x0, y0, x1, y1 = box
    contained_masks = []
    not_contained_masks = []

    for mask_dict in mask_dicts:
        segmentation = mask_dict["segmentation"]
        if not isinstance(segmentation, np.ndarray) or segmentation.dtype != bool:
            raise ValueError("The 'segmentation' key must contain a 2D boolean numpy array.")
        
        # Find coordinates of all True pixels in the mask
        rows, cols = np.where(segmentation)
        
        # Check if all True pixels fall within the box
        if np.all((x0 <= cols) & (cols <= x1) & (y0 <= rows) & (rows <= y1)):
            contained_masks.append(mask_dict)
        else:
            not_contained_masks.append(mask_dict)

    return contained_masks, not_contained_masks

def sort_masks_by_horizontal_position(mask_dicts, left_is_up):
    """
    Sort masks by their horizontal (width) position based on their centers.

    Args:
        mask_dicts (list of dict): List of dictionaries, each with a key "segmentation"
                                   containing a 2D boolean array representing the mask.
        left_is_up (bool): If True, sort masks so that the leftmost mask is first.
                           If False, sort so that the rightmost mask is first.

    Returns:
        list of dict: Sorted list of mask dictionaries.
    """

    # Calculate centers
    centers = calculate_mask_centers(mask_dicts)

    # Pair masks with their center_x values
    mask_with_centers = [
        (mask, center[0])  
        for mask, center in zip(mask_dicts, centers)
        if center[0] is not None  # Exclude masks with no valid center
    ]

    # Sort by horizontal position (x-coordinate)
    sorted_masks = sorted(
        mask_with_centers,
        key=lambda pair: pair[1],  
        reverse=not left_is_up  # Reverse for rightmost first
    )

    # Extract and return only the sorted masks
    return [mask for mask, _ in sorted_masks]

def get_border_pixels(mask, left_is_up):
    if not isinstance(mask, np.ndarray) or mask.dtype != bool:
        raise ValueError("The mask must be a 2D boolean numpy array.")
    
    # Find the border pixels
    border = np.zeros_like(mask, dtype=bool)
    if left_is_up:
        border[:, :-1] |= mask[:, :-1] & ~mask[:, 1:]  # Left edge
    else:
        border[:, 1:] |= mask[:, 1:] & ~mask[:, :-1]  # Right edge

    rows, cols = np.where(border)
    border_pixels = np.array(list(zip(cols, rows)))

    return border, np.array(border_pixels, dtype=np.uint64)

class SAM2Model:

    def __init__(self):
        checkpoint = f"{os.environ['HOME']}/repos/ckp/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=build_sam2(model_cfg, checkpoint),
            points_per_side=24,
            points_per_batch=44,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.9,
            stability_score_offset=0.7,
            # crop_n_layers=2,
            # box_nms_thresh=0.7,
            crop_n_points_downscale_factor=4,
            min_mask_region_area=25.0,
            # use_m2m=True,
        )

    def predict(self, image):
        return self.mask_generator.generate(image)
