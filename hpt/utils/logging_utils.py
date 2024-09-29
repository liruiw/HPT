import cv2
import tabulate
from tqdm import tqdm


import cv2
import numpy as np

import torch
import hydra
import csv
import einops
import os


def mkdir_if_missing(dst_dir):
    """make destination folder if it's missing"""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def module_max_param(module):
    """
    Calculate the max parameter value among all parameters in the given module.
    """    
    if hasattr(module, "module"):
        module = module.module  # data parallel

    def maybe_max(x):
        return float(torch.abs(x).max()) if x is not None else 0

    max_data = np.amax([(maybe_max(param.data)) for name, param in module.named_parameters()])
    return max_data


def module_mean_param(module):
    """
    Calculate the mean parameter value among all parameters in the given module.
    """    
    if hasattr(module, "module"):
        module = module.module  # data parallel

    def maybe_mean(x):
        return float(torch.abs(x).mean()) if x is not None else 0

    max_data = np.mean([(maybe_mean(param.data)) for name, param in module.named_parameters()])
    return max_data


def module_max_gradient(module):
    """
    Calculate the maximum gradient value among all parameters in the given module.
    """
    if hasattr(module, "module"):
        module = module.module  # data parallel

    def maybe_max(x):
        return torch.abs(x).max().item() if x is not None else 0

    max_grad = np.amax([(maybe_max(param.grad)) for name, param in module.named_parameters()])
    return max_grad


def print_and_write(file_handle, text):
    print(text)

    if file_handle is not None:
        if type(file_handle) is list:
            for f in file_handle:
                f.write(text + "\n")
        else:
            file_handle.write(text + "\n")
    return text


