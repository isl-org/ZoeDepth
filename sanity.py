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

import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import get_image_from_url, colorize
import torch

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from pprint import pprint


torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: Running on CPU. This will be slow. Check your CUDA installation.")

print("*" * 20 + " Testing zoedepth " + "*" * 20)
conf = get_config("zoedepth", "infer")


print("Config:")
pprint(conf)

model = build_model(conf).to(DEVICE)
model.eval()
x = torch.rand(1, 3, 384, 512).to(DEVICE)

print("-"*20 + "Testing on a random input" + "-"*20)

with torch.no_grad():
    out = model(x)

if isinstance(out, dict):
    # print shapes of all outputs
    for k, v in out.items():
        if v is not None:
            print(k, v.shape)
else:
    print([o.shape for o in out if o is not None])

print("\n\n")
print("-"*20 + " Testing on an indoor scene from url " + "-"*20)

# Test img
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS4W8H_Nxk_rs3Vje_zj6mglPOH7bnPhQitBH8WkqjlqQVotdtDEG37BsnGofME3_u6lDk&usqp=CAU"
img = get_image_from_url(url)
orig_size = img.size
X = ToTensor()(img)
X = X.unsqueeze(0).to(DEVICE)

print("X.shape", X.shape)
print("predicting")

with torch.no_grad():
    out = model.infer(X).cpu()

# or just, 
# out = model.infer_pil(img)


print("output.shape", out.shape)
pred = Image.fromarray(colorize(out))
# Stack img and pred side by side for comparison and save
pred = pred.resize(orig_size, Image.ANTIALIAS)
stacked = Image.new("RGB", (orig_size[0]*2, orig_size[1]))
stacked.paste(img, (0, 0))
stacked.paste(pred, (orig_size[0], 0))

stacked.save("pred.png")
print("saved pred.png")


model.infer_pil(img, output_type="pil").save("pred_raw.png")