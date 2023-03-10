# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import gradio as gr
import torch

from .gradio_depth_pred import create_demo as create_depth_pred_demo
from .gradio_im_to_3d import create_demo as create_im_to_3d_demo
from .gradio_pano_to_3d import create_demo as create_pano_to_3d_demo


css = """
#img-display-container {
    max-height: 50vh;
    }
#img-display-input {
    max-height: 40vh;
    }
#img-display-output {
    max-height: 40vh;
    }
    
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

title = "# ZoeDepth"
description = """Official demo for **ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth**.

ZoeDepth is a deep learning model for metric depth estimation from a single image.

Please refer to our [paper](https://arxiv.org/abs/2302.12288) or [github](https://github.com/isl-org/ZoeDepth) for more details."""

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Tab("Depth Prediction"):
        create_depth_pred_demo(model)
    with gr.Tab("Image to 3D"):
        create_im_to_3d_demo(model)
    with gr.Tab("360 Panorama to 3D"):
        create_pano_to_3d_demo(model)

if __name__ == '__main__':
    demo.queue().launch()