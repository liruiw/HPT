"""Device selection helpers so inference runs on XPU, CUDA or CPU from one code path.

Detection order is XPU, then CUDA, then CPU. An explicit device always wins.
"""

from typing import Union

import torch

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu`` > ``cuda`` > ``cpu``."""
    for device_type in _ACCELERATORS:
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve ``None`` to the best available device, honour anything else verbatim."""
    if device is None:
        return torch.device(default_device_type())
    return torch.device(device)


def empty_cache(device: DeviceLike = None) -> None:
    """Release cached allocator blocks of ``device``'s backend; no-op on CPU.

    ``torch.cuda.empty_cache()`` has no meaning for an XPU or CPU run, so the call
    is routed to the backend that holds the model.
    """
    device_type = resolve_device(device).type
    if device_type == "xpu" and is_available("xpu"):
        torch.xpu.empty_cache()
    elif device_type == "cuda" and is_available("cuda"):
        torch.cuda.empty_cache()
