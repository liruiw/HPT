import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import torch.nn as nn


def module_max_param(module):
    """
    Calculate the maximum absolute value of the parameters in the given module.
    """
    if hasattr(module, "module"):
        module = module.module  # data parallel

    def maybe_max(x):
        return float(torch.abs(x).max()) if x is not None else 0

    max_data = np.amax([(maybe_max(param.data)) for name, param in module.named_parameters()])
    return max_data


def module_mean_param(module):
    """
    Calculate the mean value of the absolute values of the parameters in a module.
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
