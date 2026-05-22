import sys
import rclpy

from rclpy.node import Node
from softenable_display_msgs.srv import SetDisplay

class Slideshow(Node):

    def __init__(self, n_slides = 12):
        super().__init__("Slideshow")
        
        self.n_slides = n_slides
        self.cli_display = self.create_client(SetDisplay, "/set_display")

    def start_slideshow(self, start_slide: int = 1):

        input(f"start slideshow? intial slide: {start_slide}")
        for slide_no in range(start_slide, self.n_slides+1):
            self.cli_display.call_async(SetDisplay.Request(name=f"study-{slide_no}", use_tts=True))
            
            if slide_no < self.n_slides:
                input(f"switch to slide {slide_no+1}?")
        
        print("slides done!")

if __name__ == '__main__':
    rclpy.init()
    n_slides = 12
    start_slide = 1

    try:
        start_slide = int(sys.argv[-1])
        assert 0 < start_slide <= n_slides, f"slide {start_slide} out of range!"
    except ValueError:
        print("Last argument not an int; assuming start_slide = 0")

    slides = Slideshow(n_slides)
    slides.start_slideshow(start_slide=start_slide)