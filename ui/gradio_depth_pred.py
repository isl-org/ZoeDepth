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
from zoedepth.utils.misc import colorize
from PIL import Image
import tempfile

def predict_depth(model, image):
    depth = model.infer_pil(image)
    return depth

def create_demo(model):
    gr.Markdown("### Depth Prediction demo")
    with gr.Row():
        input_image = gr.Image(label="Input Image", type='pil', elem_id='img-display-input').style(height="auto")
        depth_image = gr.Image(label="Depth Map", elem_id='img-display-output')
    raw_file = gr.File(label="16-bit raw depth, multiplier:256")
    submit = gr.Button("Submit")

    def on_submit(image):
        depth = predict_depth(model, image)
        colored_depth = colorize(depth, cmap='gray_r')
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth = Image.fromarray((depth*256).astype('uint16'))
        raw_depth.save(tmp.name)
        return [colored_depth, tmp.name]
    
    submit.click(on_submit, inputs=[input_image], outputs=[depth_image, raw_file])
    # examples = gr.Examples(examples=["examples/person_1.jpeg", "examples/person_2.jpeg", "examples/person-leaves.png", "examples/living-room.jpeg"],
    #                        inputs=[input_image])