from stack_msgs.srv import SAM2Srv, DINOSrv

import time
import rclpy
from rclpy.node import Node

from stack_detect.helpers.sam2_model import SAM2Model
from stack_detect.helpers.dino_model import DINOModel, plot_boxes_to_image

class Sam2DINOSrv(Node):

    def __init__(self):
        super().__init__('Sam2DINOSrv')

        self.declare_parameter('dino_cpu', False)
        self.dino_cpu = self.get_parameter("dino_cpu").get_parameter_value().bool_value

        self.srv = self.create_service(SAM2Srv, 'sam2_srv', self.sam2_callback)
        self.srv = self.create_service(DINOSrv, 'dino_srv', self.dino_callback)


        self.get_logger().info("sam2 setup done!")

        self.sam = SAM2Model()
        self.dino = DINOModel(cpu_only=self.dino_cpu)
            
    def sam2_callback(self, request, response):
        
        
        return response

    def dino_callback(self, request, response):
        print("got DINO request!")
        
        img_raw = request.image
        image_pil = Image.fromarray(img_raw, mode="RGB")

        #### Run DINO
        dino_start = time.time()
        self.get_logger().info("running DINO ...")
        boxes_px, pred_phrases, confidences = self.dino.predict(
            image_pil, 
            "detect all stacks of clothing"
        )
        self.get_logger().info(f"DINO took {round(time.time()-dino_start,2)}s")
        
        image_with_box = plot_boxes_to_image(image_pil.copy(), boxes_px, pred_phrases)[0]

        response.image = image_with_box
        response.boxes_px = boxes_px

        return response


def main():
    rclpy.init()

    minimal_service = Sam2DINOSrv()

    rclpy.spin(minimal_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()