import os

from stack_detect.helpers.dino import load_model, get_grounding_output, plot_boxes_to_image

class DINOModel:

    def __init__(self, cpu_only=False):
        self.cpu_only = cpu_only
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.token_spans = None
        prefix = f"{os.environ['HOME']}/repos/"
        self.model = load_model(prefix+"GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", prefix+"ckp/groundingdino_swint_ogc.pth", cpu_only=self.cpu_only)

    def predict(self, image_pil, prompt):
        return get_grounding_output(
            self.model, 
            image_pil, 
            prompt, 
            self.box_threshold, 
            self.text_threshold, 
            cpu_only=self.cpu_only, 
            token_spans=eval(f"{self.token_spans}")
        )
        