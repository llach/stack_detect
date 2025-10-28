import cv2
import numpy as np


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


def get_bag_pose_from_array(img_array, point_offset=0.15, show_debug=False):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found in cropped image")

    contour_info = [(c, cv2.boundingRect(c)) for c in contours]
    contour_info.sort(key=lambda x: x[1][2] * x[1][3], reverse=True)
    top_contour = contour_info[0][0]

    rect = cv2.minAreaRect(top_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    (cx, cy), (w, h), angle = rect

    if w < h:
        rot_angle = angle
    else:
        rot_angle = angle + 90

    angle_to_vertical = (90 + rot_angle) % 180

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
    offset_point = top_left + point_offset * width_vec + 0.5 * height_vec
    offset_point = tuple(map(int, offset_point))

    if show_debug:
        vis = img_array.copy()
        cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)
        cv2.circle(vis, offset_point, 6, (0, 255, 255), -1)
        cv2.putText(vis, f"{angle_to_vertical:.2f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Pose Debug", vis)
        cv2.waitKey(1)

    return angle_to_vertical, box, offset_point



