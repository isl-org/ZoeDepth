import torch
from PIL import Image
from cog import BasePredictor, Input, Path

from zoedepth.utils.misc import get_image_from_url, colorize


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.models = {
            k: torch.hub.load(".", k, source="local", pretrained=True).to("cuda")
            for k in ["ZoeD_N", "ZoeD_K", "ZoeD_NK"]
        }

    def predict(
        self,
        model_type: str = Input(
            default="ZoeD_N",
            choices=["ZoeD_N", "ZoeD_K", "ZoeD_NK"],
        ),
        image: Path = Input(description="Input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        zoe = self.models[model_type]
        image = Image.open(image).convert("RGB")  # load
        # depth_numpy = zoe.infer_pil(image)  # as numpy

        depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

        # Colorize output
        colored = colorize(depth_tensor)

        # save colored output
        output_path = "/tmp/out.png"
        Image.fromarray(colored).save(output_path)

        return Path(output_path)
