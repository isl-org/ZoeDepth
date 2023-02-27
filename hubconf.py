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

dependencies=['torch']
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model
import numpy as np
import torch


# ZoeD_N
def ZoeD_N(pretrained=False, midas_model_type="DPT_BEiT_L_384", config_mode="infer", **kwargs):
    """Zoe_M12_N model. This is the version of ZoeDepth that has a single metric head
    Args:
        pretrained (bool): If True, returns a model pre-trained on NYU-Depth-V2
        midas_model_type (str): Midas model type. Should be one of the models as listed in torch.hub.list("intel-isl/MiDaS"). Default: DPT_BEiT_L_384
        config_mode (str): Config mode. Should be one of "infer", "train" or "eval". Default: "infer"
    
    Keyword Args:
        **kwargs: Additional arguments to pass to the model
        The following arguments are supported:
            train_midas (bool): If True, returns a model that with trainable midas base. Default: False
            use_pretrained_midas (bool): If True, returns a model that uses pretrained midas base. Default: False
            n_bins (int): Number of bin centers. Defaults to 64.
            bin_centers_type (str): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int): bin embedding dimension. Defaults to 128.
            min_depth (float): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int]): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 1000.
            attractor_gamma (int): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str): Attraction aggregation "sum" or "mean". Defaults to 'mean'.
            attractor_type (str): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'inv'.
            min_temp (int): Lower bound for temperature of output probability distribution. Defaults to 0.0212.
            max_temp (int): Upper bound for temperature of output probability distribution. Defaults to 50.
            force_keep_ar (bool): If True, the model will keep the aspect ratio of the input image. Defaults to True.
    """
    if pretrained and midas_model_type != "DPT_BEiT_L_384":
        raise ValueError(f"Only DPT_BEiT_L_384 MiDaS model is supported for pretrained Zoe_N model, got: {midas_model_type}")

    if not pretrained:
        pretrained_resource = None
    else:
        pretrained_resource = "url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt"

    config = get_config("zoedepth", config_mode, pretrained_resource=pretrained_resource, **kwargs)
    model = build_model(config)
    return model

# ZoeD_K
def ZoeD_K(pretrained=False, midas_model_type="DPT_BEiT_L_384", config_mode="infer", **kwargs):
    """Zoe_M12_K model. This is the version of ZoeDepth that has a single metric head
    Args:
        pretrained (bool): If True, returns a model pre-trained on NYU-Depth-V2
        midas_model_type (str): Midas model type. Should be one of the models as listed in torch.hub.list("intel-isl/MiDaS"). Default: DPT_BEiT_L_384
        config_mode (str): Config mode. Should be one of "infer", "train" or "eval". Default: "infer"
    
    Keyword Args:
        **kwargs: Additional arguments to pass to the model
        The following arguments are supported:
            train_midas (bool): If True, returns a model that with trainable midas base. Default: False
            use_pretrained_midas (bool): If True, returns a model that uses pretrained midas base. Default: False
            n_bins (int): Number of bin centers. Defaults to 64.
            bin_centers_type (str): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int): bin embedding dimension. Defaults to 128.
            min_depth (float): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int]): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 1000.
            attractor_gamma (int): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str): Attraction aggregation "sum" or "mean". Defaults to 'mean'.
            attractor_type (str): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'inv'.
            min_temp (int): Lower bound for temperature of output probability distribution. Defaults to 0.0212.
            max_temp (int): Upper bound for temperature of output probability distribution. Defaults to 50.
            force_keep_ar (bool): If True, the model will keep the aspect ratio of the input image. Defaults to True.

    """
    if pretrained and midas_model_type != "DPT_BEiT_L_384":
        raise ValueError(f"Only DPT_BEiT_L_384 MiDaS model is supported for pretrained Zoe_K model, got: {midas_model_type}")
    
    if not pretrained:
        pretrained_resource = None
    else:
        pretrained_resource = "url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_K.pt"

    config = get_config("zoedepth", config_mode, pretrained_resource=pretrained_resource, config_version="kitti", **kwargs)
    model = build_model(config)
    return model

# Zoe_NK
def ZoeD_NK(pretrained=False, midas_model_type="DPT_BEiT_L_384", config_mode="infer", **kwargs):
    """ZoeDepthNK model. This is the version of ZoeDepth that has two metric heads and uses a learned router to route to experts.
    Args:
        pretrained (bool): If True, returns a model pre-trained on NYU-Depth-V2
        midas_model_type (str): Midas model type. Should be one of the models as listed in torch.hub.list("intel-isl/MiDaS"). Default: DPT_BEiT_L_384
    
    Keyword Args:
        **kwargs: Additional arguments to pass to the model
        The following arguments are supported:
            train_midas (bool): If True, returns a model that with trainable midas base. Defaults to True
            use_pretrained_midas (bool): If True, returns a model that uses pretrained midas base. Defaults to True
            bin_conf (List[dict]): A list of dictionaries that contain the bin configuration for each metric head. Each dictionary should contain the following keys: 
                                    "name" (str, typically same as the dataset name), "n_bins" (int), "min_depth" (float), "max_depth" (float)
                                   The length of this list determines the number of metric heads.
            bin_centers_type (str): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int): bin embedding dimension. Defaults to 128.

            n_attractors (List[int]): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 1000.
            attractor_gamma (int): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str): Attraction aggregation "sum" or "mean". Defaults to 'mean'.
            attractor_type (str): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'inv'.

            min_temp (int): Lower bound for temperature of output probability distribution. Defaults to 0.0212.
            max_temp (int): Upper bound for temperature of output probability distribution. Defaults to 50.
            
            memory_efficient (bool): Whether to use memory efficient version of attractor layers. Memory efficient version is slower but is recommended incase of multiple metric heads in order save GPU memory. Defaults to True.

    """
    if pretrained and midas_model_type != "DPT_BEiT_L_384":
        raise ValueError(f"Only DPT_BEiT_L_384 MiDaS model is supported for pretrained Zoe_NK model, got: {midas_model_type}")
    
    if not pretrained:
        pretrained_resource = None
    else:
        pretrained_resource = "url::https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt"

    config = get_config("zoedepth_nk", config_mode, pretrained_resource=pretrained_resource, **kwargs)
    model = build_model(config)
    return model