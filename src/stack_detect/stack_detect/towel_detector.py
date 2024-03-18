"""Subscriber module"""
import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge 
from rclpy.parameter import Parameter

from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image
from rcl_interfaces.msg import SetParametersResult

class ColorLineDetector:

    def __init__(self, hue=330, hue_tol=0.06, sat_range=(40,255), offset=(0,0), morph_size=15):
        self.hue = hue
        self.hue_tol = hue_tol
        self.offset = np.array(offset).astype(np.int16)
        self.sat_range = sat_range
        self.morph_size = morph_size

    def detect(self, frame):
        # Step 1: filter for color (uses HSL space for better stability)
        towel_col = self._filter_hue_color(frame)
        towel_bin = np.where(towel_col == [0,0,0], [0,0,0], [255,255,255]).astype("uint8")

        upper_col = np.hstack([frame, towel_col])

        # Step 2: 1st opening to remove outliers; 2nd closing to have a consistent towel blob
        morph_open = cv2.morphologyEx(towel_bin, cv2.MORPH_OPEN, np.ones((self.morph_size, self.morph_size),np.uint8))
        morph_close = cv2.morphologyEx(morph_open, cv2.MORPH_CLOSE, np.ones((self.morph_size, self.morph_size),np.uint8))

        # Step 2.1: TODO - add filter for patch size!

        # Step 3: determine the lowest non-black pixel for columns that have at least N=1 non-black pixel
        # this assumes that the camera is not rotated more than 90deg w.r.t. the ground
        # this might be even more stable if we choose to ignore columns with too few non-black pixels (comparison in the line that continues the loop)
        final_img = frame[:]
        line_img = np.ones_like(towel_col)
        line_bin_img = np.zeros(towel_col.shape[:2]).astype(np.uint8)
        line_pxls = []
        for i in range(morph_close.shape[1]):
            idxs, = np.where(np.all(morph_close[:,i,:] != [0,0,0], axis=1)) # get indices of all non-black pixels
            if len(idxs)==0: continue # → no towel pixels detected in this column
            line_img[max(idxs), i] = [255,255,255] # "winner" is painted
            final_img[max(idxs), i] = [255,255,255] # "winner" is painted
            line_bin_img[max(idxs), i] = 1
            line_pxls.append([max(idxs), i])

        lower_col = np.hstack([morph_close, line_img])

        # Step 4: perform Hough transform to extract lines
        hough_img = line_img[:]
        lines = cv2.HoughLines(line_bin_img, 5, np.pi/180, 80)
        if lines is not None: self._draw_hough_line(hough_img, lines[0])

        upper_col = np.hstack([upper_col, hough_img])
        lower_col = np.hstack([lower_col, final_img])
        all_steps_img = np.vstack([upper_col, lower_col])

        # Step 5: grasp generation
        if len(line_pxls) > 0:
            gripper_target = np.mean(line_pxls, axis=0).astype(np.int16)
            center = np.array(final_img.shape[:2])/2
            center = center.astype(np.int16)+self.offset
            (y_err, x_err) = np.round(center-gripper_target, 1)

            cv2.putText(final_img, f"X err: {x_err}", (20,400), 1, 2, (255, 50, 50), thickness=2)
            cv2.putText(final_img, f"Y err: {y_err}", (20,440), 1, 2, (255, 50, 50), thickness=2)
            cv2.circle(final_img, gripper_target[::-1], 5, (0, 0, 150), 4)
            cv2.circle(final_img, center[::-1], 5, (0,255,0), 4)
            cv2.line(final_img, center[::-1], gripper_target[::-1], (122, 122, 122), 1)

            return (center, gripper_target), (x_err, y_err), (all_steps_img,  final_img)
        else:
            cv2.putText(final_img, f"-- no line found --", (20,440), 1, 2, (255, 50, 50), thickness=2)
            return None, None, (all_steps_img,  final_img)

    # tol in [0,1] is tolerance in percent
    def _filter_hue_color(self, img):
        tol = self.hue_tol
        target_hue = self.hue/2 # cv stores H/2 in 8bit images

        # convert image to HLS color space and separate individual dimensions
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hue, sat, _ = hls[:,:,0], hls[:,:,1], hls[:,:,2]

        # create filters for hue and saturation
        hue_mask = cv2.inRange(hue, (1-tol)*target_hue, (1+tol)*target_hue)
        sat_mask = cv2.inRange(sat, *self.sat_range)
        mask = hue_mask & sat_mask

        # keep pixels that match the mask, else black
        return cv2.bitwise_and(img, img, mask=mask)

    def _draw_hough_line(self, img, line, color=(255,0,0)):
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1,y1), (x2,y2), color, 2)
            

class TowelDetector(Node):
    """Subscriber node"""

    def __init__(self):
        super().__init__("towel_detector")

        # defaults                                          pink        yellow
        self.declare_parameter("debug_img", "none")
        self.declare_parameter("hue", 48)              #   340             48       
        self.declare_parameter("hue_tol", 0.1)         #    0.04           0.1
        self.declare_parameter("sat_range", [120, 255]) #           [120, 255]
        self.declare_parameter("offset", [0, 0])        #
        self.declare_parameter("morph_size", 4)        #    15             4

        self.debug_img = self.get_parameter("debug_img").value
        self.hue = self.get_parameter("hue").value
        self.hue_tol = self.get_parameter("hue_tol").value
        self.sat_range = self.get_parameter("sat_range").value
        self.offset = self.get_parameter("offset").value
        self.morph_size = self.get_parameter("morph_size").value

        assert self.debug_img in ["final", "all", "none"], 'self.debug_img in ["final", "all", None]'

        self.add_on_set_parameters_callback(self.parameter_callback)

        self.subscription = self.create_subscription(
            Image, "/video_in", self.listener_callback, 0
        )

        self.err_pub = self.create_publisher(Int16MultiArray, 'line_img_error', 10)

        if self.debug_img:
            self.img_pub = self.create_publisher(Image, "debug_img", 10)

        self.br = CvBridge()
        self.ld = ColorLineDetector()
        self.set_detector_params()

        print("towel detection running")

    def set_detector_params(self):
        self.ld.hue = self.hue
        self.ld.hue_tol = self.hue_tol
        self.ld.sat_range = self.sat_range
        self.ld.offset = self.offset
        self.ld.morph_size = self.morph_size

    def parameter_callback(self, params):
        for p in params:
            if hasattr(self, p.name): 
                print(f"setting {p.name} {p.value}")
                setattr(self, p.name, p.value)
        self.set_detector_params()
        return SetParametersResult(successful=True)

    def listener_callback(self, msg):
        """Callback function when message is received"""
        # self.get_logger().info(f'image size {msg.width}x{msg.height}')
        cam_img = self.br.imgmsg_to_cv2(msg, "bgr8")

        # detect stack and determine pixel-space errors
        _, err, (all_img, final_img) = self.ld.detect(cam_img)
        if err is not None: self.err_pub.publish(Int16MultiArray(data=err))

        if self.debug_img:
            dbg_img = all_img if self.debug_img == "all" else final_img

            # cv2 imshow for debugging
            # cv2.imshow("", dbg_img)
            # cv2.waitKey(50)

            self.img_pub.publish(self.br.cv2_to_imgmsg(dbg_img, encoding="bgr8"))

    

def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    towel_detector = TowelDetector()

    rclpy.spin(towel_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    towel_detector.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
