import cv2
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------
# --- Utility functions -----------------------------------------
# ---------------------------------------------------------------
def expand_bounding_box(box, image_width, image_height, scale=1.1):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = (x1 - x0) * scale
    h = (y1 - y0) * scale
    x0 = int(cx - w / 2)
    y0 = int(cy - h / 2)
    x1 = int(cx + w / 2)
    y1 = int(cy + h / 2)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(image_width, x1)
    y1 = min(image_height, y1)
    return x0, y0, x1, y1


def get_bag_pose_from_array(
    img_array,
    point_offset=0.15,
    gauss_size=7,
    canny_thresh=(50, 150),
    closing_kernel_size=30
):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # --- Preprocessing ---
    blur = cv2.GaussianBlur(gray, (gauss_size, gauss_size), 0)
    edges = cv2.Canny(blur, *canny_thresh)

    # --- Small morphological closing ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # --- Contours ---
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in cropped image")

    contour_info = [(c, cv2.boundingRect(c)) for c in contours]
    contour_info.sort(key=lambda x: x[1][2] * x[1][3], reverse=True)
    top_contour = contour_info[0][0]

    # --- Rotated bounding box ---
    rect = cv2.minAreaRect(top_contour)
    box = cv2.boxPoints(rect).astype(int)
    (cx, cy), (w, h), angle = rect

    # --- Offset point computation ---
    rect_pts = np.array(cv2.boxPoints(rect))
    rect_pts = rect_pts[np.lexsort((rect_pts[:, 1], rect_pts[:, 0]))]
    left_pts = rect_pts[:2]
    right_pts = rect_pts[2:]
    left_pts = left_pts[np.argsort(left_pts[:, 1])]
    right_pts = right_pts[np.argsort(right_pts[:, 1])]
    top_left, bottom_left = left_pts
    top_right, bottom_right = right_pts

    width_vec = top_right - top_left
    height_vec = bottom_left - top_left
    offset_point = top_left + 0.5 * width_vec + point_offset * height_vec
    offset_point = tuple(map(int, offset_point))

    # --- Build debug visualization (if enabled) ---
    vis_tl = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    vis_tr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    vis_bl = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    vis_br = img_array.copy()
    cv2.drawContours(vis_br, [top_contour], -1, (0, 0, 255), 2)
    cv2.drawContours(vis_br, [box], 0, (255, 255, 0), 1)
    cv2.circle(vis_br, offset_point, 6, (0, 255, 255), -1)
    cv2.putText(vis_br, f"{angle:.2f} deg", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    top_row = np.hstack([vis_tl, vis_tr])
    bottom_row = np.hstack([vis_bl, vis_br])
    vis = np.vstack([top_row, bottom_row])

    cv2.putText(vis, datetime.now().strftime('%H:%M:%S.%f')[:-3], (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return angle, box, offset_point, top_contour, vis


