import os
import cv2
import time
import glob
import rclpy
import threading

from PIL import Image
from threading import Lock
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import CompressedImage

from datetime import datetime

class ClassifyCollector(Node):

    def __init__(self):
        super().__init__("ClassifyCollector")
        self.log = self.get_logger()

        self.bridge = CvBridge()
        self.cbg = ReentrantCallbackGroup()

        self.img_lock = Lock()
        self.img_msg = None

        from rclpy.qos import qos_profile_sensor_data

        self.img_sub = self.create_subscription(
            CompressedImage,
            "/camera/color/image_raw/compressed",
            self.rgb_cb,
            qos_profile=qos_profile_sensor_data,
            callback_group=self.cbg
        )

        
    def rgb_cb(self, msg):
        with self.img_lock:
            self.img_msg = msg


def collect(node):
    node.get_logger().info("waiting for data ...")
    # Wait until at least one image has been received.
    while node.img_msg is None:
        time.sleep(0.05)
    
    classname = input("classname? ")
    outdir = f"{os.environ['HOME']}/repos/unstack_classify/{classname}"
    print("storing in", outdir)
    
    os.makedirs(outdir, exist_ok=True)
    
    n_imgs = len(glob.glob1(outdir, "*.png"))
    print(f"got {n_imgs} images already")
    while True:
        inp = input("record? ")
        if inp.lower().strip() == "q":
            break

        # Wait for a *new* image to arrive:
        with node.img_lock:
            last_msg = node.img_msg
            # Clear the stored message so we can wait for a new one.
            node.img_msg = None

        # Wait until a new message comes in.
        while True:
            with node.img_lock:
                if node.img_msg is not None and node.img_msg != last_msg:
                    current_msg = node.img_msg
                    break
            time.sleep(0.01)

        # Now convert the newly received image.
        img = node.bridge.compressed_imgmsg_to_cv2(current_msg)
        img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_raw, mode="RGB")
        
        filename = f"{datetime.now().strftime('%H.%M.%S.%f_%d.%m.%Y')}.png"
        image_pil.save(f"{outdir}/{filename}")
        
        n_imgs += 1
        print(f"#{n_imgs} | {filename}")
    
    print("bye")


def main(args=None):
    """Creates subscriber node and spins it"""
    rclpy.init(args=args)

    node = ClassifyCollector()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
     # Start the executor in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        collect(node)
    except KeyboardInterrupt:
        print("Keyboard Interrupt!")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
