import os
import cv2
import numpy as np

from datetime import datetime
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from stack_approach.helpers import pixel_to_point

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
        color_mask = np.random.random(3)*[255,0,255] # less red so if we draw over masks, we have higher contrast
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    return img.astype(np.uint8)

def find_masks_in_box(mask_dicts, box, thresh=0.95):
    """
    Find masks that are fully contained within a bounding box, partially contained,
    and masks that are not.

    Args:
        mask_dicts (list of dict): List of dictionaries, each with a key "segmentation"
                                   containing a 2D boolean array representing the mask.
        box (tuple): Bounding box in pixel space (x0, y0, x1, y1).
        thresh (float): Minimum fraction of mask pixels that must be inside the box
                        for it to be considered contained. Defaults to 0.95.

    Returns:
        tuple: (contained_masks, not_contained_masks)
               - contained_masks: List of dicts for masks contained in the box
                 based on the threshold.
               - not_contained_masks: List of dicts for masks not meeting the threshold.
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
        total_pixels = len(rows)  # Total number of True pixels in the mask

        if total_pixels == 0:  # Skip empty masks
            not_contained_masks.append(mask_dict)
            continue

        # Check how many True pixels fall within the box
        in_box = (x0 <= cols) & (cols <= x1) & (y0 <= rows) & (rows <= y1)
        pixels_in_box = np.sum(in_box)

        # Calculate the fraction of pixels inside the box
        fraction_in_box = pixels_in_box / total_pixels

        # Classify based on threshold
        if fraction_in_box >= thresh:
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
    """
    Extracts the border pixels of a binary mask. Only one pixel per row is selected, 
    either the leftmost or rightmost pixel based on the `left_is_up` parameter.

    Args:
        mask (np.ndarray): A 2D binary boolean mask.
        left_is_up (bool): If True, return the leftmost pixel per row. 
                           If False, return the rightmost pixel per row.

    Returns:
        tuple: (binary_mask, pixel_array)
               - binary_mask (np.ndarray): A 2D binary mask with border pixels set to True.
               - pixel_array (np.ndarray): Array of border pixel positions as [[x1, y1], [x2, y2], ...].
    """
    if not isinstance(mask, np.ndarray) or mask.dtype != bool:
        raise ValueError("The mask must be a 2D boolean numpy array.")
    
    border_mask = np.zeros_like(mask, dtype=bool)  # Initialize an empty binary mask
    border_pixels = []

    # Iterate over each row to find the leftmost or rightmost pixel
    for y in range(mask.shape[0]):
        row = mask[y, :]
        if np.any(row):  # Check if the row contains any True values
            if not left_is_up:
                x = np.argmax(row)  # Index of the first True value (leftmost)
            else:
                x = len(row) - 1 - np.argmax(row[::-1])  # Index of the last True value (rightmost)
            border_mask[y, x] = True  # Mark the pixel in the binary mask
            border_pixels.append([x, y])

    return border_mask, np.array(border_pixels, dtype=np.uint64)


def filter_masks_by_size(mask_dicts, box, min_width=0.7, max_height=0.6, rotated=False):
    """
    Filter masks based on their width and height as fractions of the bounding box size.

    Args:
        mask_dicts (list of dict): List of dictionaries, each with a key "segmentation"
                                   containing a 2D boolean array representing the mask.
        box (tuple): Bounding box in pixel space (x0, y0, x1, y1).
        min_width (float): Minimum width fraction (default: 0.7).
        max_height (float): Maximum height fraction (default: 0.6).
        rotated (bool): If True, the width dimension is treated as the image's y axis,
                        otherwise as the image's x axis.

    Returns:
        tuple: (masks_meeting_criteria, masks_not_meeting_criteria)
               - masks_meeting_criteria: List of dicts for masks meeting the criteria.
               - masks_not_meeting_criteria: List of dicts for masks not meeting the criteria.
    """
    x0, y0, x1, y1 = box
    box_width = y1 - y0 if rotated else x1 - x0
    box_height = x1 - x0 if rotated else y1 - y0

    masks_meeting_criteria = []
    masks_not_meeting_criteria = []

    for mask_dict in mask_dicts:
        segmentation = mask_dict["segmentation"]
        if not isinstance(segmentation, np.ndarray) or segmentation.dtype != bool:
            raise ValueError("The 'segmentation' key must contain a 2D boolean numpy array.")
        
        # Find the bounding box of the mask
        rows, cols = np.where(segmentation)
        if len(rows) == 0 or len(cols) == 0:  # Empty mask
            masks_not_meeting_criteria.append(mask_dict)
            continue

        mask_width = (rows.max() - rows.min() + 1) if rotated else (cols.max() - cols.min() + 1)
        mask_height = (cols.max() - cols.min() + 1) if rotated else (rows.max() - rows.min() + 1)

        # Normalize by box dimensions
        normalized_width = mask_width / box_width
        normalized_height = mask_height / box_height

        # Check if mask meets the criteria
        if normalized_width >= min_width and normalized_height <= max_height:
            masks_meeting_criteria.append(mask_dict)
        else:
            masks_not_meeting_criteria.append(mask_dict)

    return masks_meeting_criteria, masks_not_meeting_criteria

class SAM2Model:

    def __init__(self, 
        points_per_side = 24, 
        points_per_batch = 44, 
        pred_iou_thresh = 0.7,
        stability_score_thresh=0.8,
        stability_score_offset=0.7
    ):
        checkpoint = f"{os.environ['HOME']}/repos/ckp/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=build_sam2(model_cfg, checkpoint),
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100.0,
            use_m2m=True,
        )

    def predict(self, image):
        return self.mask_generator.generate(image)

    @staticmethod
    def detect_stack(image, masks, box, mask_w=0.4):
        ### draw masks
        img_anns = draw_anns(masks)
        img_overlay = np.clip((1-mask_w)*image + mask_w*img_anns, 0, 255).astype(np.uint8)

        ### draw DINO box
        x0, y0, x1, y1 = box
        cv2.rectangle(img_overlay, (x0, y0), (x1, y1), (255, 0, 255), 2)

        ### sort masks based on horizontal center position
        sorted_masks = sort_masks_by_horizontal_position(masks, left_is_up=True)

        ### find masks where EVERY pixel falls inside the DINO box, and all also those outside
        masks_inside, masks_outside = find_masks_in_box(sorted_masks, box, thresh=.5)

        ### we discard masks that are not wide enough (small features in the background) and too tall (sometimes the stack itself is detected as a whole)
        masks_inside_ok, masks_inside_not_ok = filter_masks_by_size(masks_inside, box, rotated=True)

        ### select the first cluster, get line pixels and a mask for drawing
        if len(masks_inside_ok) > 0:
            upper_layer_mask = masks_inside_ok[0]
            line_mask, line_pixels = get_border_pixels(upper_layer_mask["segmentation"], left_is_up=True)
            line_center = np.mean(line_pixels, axis=0).astype(np.uint64)

            # draw line and grasp center
            img_overlay[line_mask] = [0,255,0]
            cv2.circle(img_overlay, line_center, 3, (100,100,100), -1)
        else:
            line_pixels = []
            line_center = None

        # draw mask centers, the ones inside are also annotated by their index from sorting
        centers_inside = calculate_mask_centers(masks_inside_ok)
        for i, c in enumerate(centers_inside):
            cv2.putText(img_overlay,f'{i}', c, cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,0),2,cv2.LINE_AA)
            cv2.circle(img_overlay, c, 2, (0,255,0), -1)

        centers_inside_not_ok = calculate_mask_centers(masks_inside_not_ok)
        for c in centers_inside_not_ok:
            cv2.circle(img_overlay, c, 2, (255,0,0), -1)

        centers_outside = calculate_mask_centers(masks_outside)
        for c in centers_outside:
            cv2.circle(img_overlay, c, 2, (255,255,255), -1)

        cv2.putText(img_overlay, datetime.now().strftime('%H:%M:%S.%f')[:-3], (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img_overlay, line_pixels, line_center
    
    @staticmethod
    def get_center_point(pixel, depth_img, K):
        # get distance, convert to 3D point
        line_dist = depth_img[pixel[1], pixel[0]]/1000 # convert to meters
        return pixel_to_point(pixel, line_dist, K) # TODO reverse pixel order here too?
