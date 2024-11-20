import cv2
import torch.nn.functional as F
import torch.nn as nn
import hydra
import torch


import hydra
from omegaconf import OmegaConf
import os
from PIL import Image
import numpy as np

from transformers import CLIPTextModel, CLIPVisionModel
from transformers import T5Tokenizer, T5Model, AutoTokenizer
from transformers import AutoImageProcessor, Dinov2Model, AutoProcessor

import gc
from ..models.policy_stem import vit_base_patch16, ResNet
import einops
import json
import tabulate


# global model cache
global_vision_model = None  # to be assigned
global_language_model = None  # to be assigned
global_vision_processor = None # to be assigned
global_language_processor = None # to be assigned


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_args_hydra(path, cfg, verbose=True):
    """
    Save the arguments and configuration to the specified path.

    Args:
        path (str): The path to save the arguments and configuration.
        cfg (omegaconf.DictConfig): The configuration object.
        verbose (bool, optional): Whether to print the configuration dictionary. Defaults to True.
    """
    mkdir_if_missing(path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    if verbose:
        print(cfg_dict)

    save_args_json(path, cfg_dict)

    with open(os.path.join(path, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)


def save_args_json(path, args, convert_to_vars=False):
    """
    Save the arguments to a JSON file.

    Args:
        path (str): The path to save the JSON file.
        args (object): The arguments to be saved.
        convert_to_vars (bool, optional): Whether to convert the arguments to variables. Defaults to False.
    """
    mkdir_if_missing(path)
    arg_json = os.path.join(path, "config.json")
    with open(arg_json, "w") as f:
        if convert_to_vars:
            args = vars(args)
        json.dump(args, f, indent=4, sort_keys=True)


def select_task(task_name_list):
    """
    Selects and returns the task configurations for the given task names.
    """
    task_configs = []
    for task_name in task_name_list:
        # assert task_name in ALL_TASK_CONFIG
        for task_config in ALL_TASK_CONFIG:
            if task_config[0] == task_name:
                task_configs.append(task_config)
                break

    return task_configs


class EinOpsRearrange(nn.Module):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)


def mkdir_if_missing(dst_dir):
    """make destination folder if it's missing"""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def get_scheduler(
    schduler_spec,
    optimizer,
    **kwargs,
):
    """
    network optimizers
    """
    sch = hydra.utils.instantiate(schduler_spec, optimizer)
    return sch


def get_optimizer(
    optimizer_spec,
    policy,
    optimizer_extra=None,
    **kwargs,
):
    """
    initializer network optimizer
    """
    trunk_params = [v for k, v in policy.named_parameters() if "trunk" in k]
    nontrunk_params = [v for k, v in policy.named_parameters() if "trunk" not in k]
    params = [
        {"params": trunk_params},
        {"params": nontrunk_params, "lr": optimizer_spec.lr * optimizer_extra.nontrunk_lr_scale},
    ]

    opt_i = eval(optimizer_spec["_target_"])(
        params=params,
        **{k: v for k, v in optimizer_spec.items() if k != "_target_"},
    )
    return opt_i


def dict_apply(x, func):
    """
    Apply a function to all values in a dictionary recursively.

    Args:
        x (dict or any): The dictionary or value to apply the function to.
        func (function): The function to apply.

    Returns:
        dict or any: The resulting dictionary or value after applying the function.
    """
    dict_type = type(x)
    if type(x) is not dict_type:
        return func(x)

    result = dict_type()
    for key, value in x.items():
        if isinstance(value, dict_type):
            result[key] = dict_apply(value, func)
        else:
            try:
                result[key] = func(value)
            except:
                result[key] = value  # fallback
    return result


def set_seed(seed):
    # set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_sinusoid_encoding_table(position_start, position_end, d_hid):
    """Sinusoid position encoding table"""

    d_vec = (1. / torch.pow(10000, 2 * (torch.arange(d_hid) / 2).floor_() / d_hid)).unsqueeze(0).float()

    sinusoid_table = torch.arange(position_start, position_end).unsqueeze(1) * d_vec

    sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return sinusoid_table.unsqueeze(0)

def unnormalize_image_numpy(image):
    """
    Unnormalizes an image in numpy format.

    Args:
        image (numpy.ndarray): The input image in numpy format.

    Returns:
        numpy.ndarray: The unnormalized image.

    Shape:
        - Input: (C, H, W)
        - Output: (C, H, W)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    image = image * std + mean
    image = image * 255
    image = image.astype(np.uint8)
    return image


def normalize_image_numpy(image: np.ndarray, resize: bool = True) -> np.ndarray:
    """
    Normalize an image in numpy format.

    Args:
        image (numpy.ndarray): The input image in H x W x 3 (uint8) format.

    Returns:
        numpy.ndarray: The normalized image in 3 x H x W format.

    Notes:
        - The input image is resized to (224, 224) using cv2.resize.
        - The image is normalized using the mean and standard deviation values from the ImageNet dataset.
        - The resulting image is transposed to have dimensions of 3 x H x W.
    """
    if resize:
        image = cv2.resize(image, (224, 224))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image / 255.0

    # convert to array
    image = np.asarray(image)

    # normalize
    image = (image - mean) / std
    return image.transpose(2, 0, 1)


def dict_apply_device(x, device):
    """
    Apply the specified device to all tensors in a nested dictionary.

    Args:
        x (dict): The nested dictionary to apply the device to.
        device (torch.device): The device to apply.

    Returns:
        dict: The nested dictionary with tensors moved to the specified device.
    """
    if type(x) is not dict:
        return value.to(device)

    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply_device(value, device)
        else:
            result[key] = value.to(device)
    return result

def download_from_huggingface(huggingface_repo_id: str):
    import huggingface_hub

    folder = huggingface_hub.snapshot_download(huggingface_repo_id)
    return folder

def get_image_embeddings(image, encoder, language=None, device="cuda", downsample=False, **kwargs):
    """
    Get embeddings for an image using the specified encoder.

    Args:
        image: The input image.
        encoder: The type of encoder to use for generating embeddings.
        language: The language used for encoding (optional).
        device: The device to use for computation (default is "cuda").
        downsample: Whether to downsample the image (default is False).
        **kwargs: Additional keyword arguments.

    Returns:
        The embeddings for the input image.

    Raises:
        Exception: If the specified encoder is not supported.
    """
    if encoder == "resnet":
        return get_resnet_embeddings(image, device=device, downsample=downsample)
    if encoder == "vit":
        return get_vit_embeddings(image, device=device)
    if encoder == "clip":
        return get_clip_embeddings(image, device=device)
    if encoder == "dino":
        return get_dino_embeddings(image, device=device)
    if encoder == "r3m":
        return get_r3m_embeddings(image, device=device)
    if encoder == "voltron":
        return get_voltron_embeddings(image, device=device)
    raise Exception("missing embedding type", encoder)

def tokenize_language(sentence: str):
    """
    Tokenize a sentence using the default tokenizer.
    """
    global global_language_processor
    if global_language_processor is None:
        global_language_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    token_ids = global_language_processor(sentence, return_tensors="pt")['input_ids'].detach().cpu().numpy()
    return token_ids[0]

@torch.no_grad()
def get_dino_embeddings(image, device="cuda", image_token_size=(3, 3)):
    """Get DINO embedding."""
    global global_vision_model, global_vision_processor
    if global_vision_model is None:
        global_vision_model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        global_vision_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        global_vision_model = global_vision_model.to(device)

    # vision
    image = Image.fromarray(image)
    vision_inputs = global_vision_processor(images=image, return_tensors="pt").to(device)
    vision_outputs = global_vision_model(**vision_inputs)

    # 16 -> 50 (7*7+1)
    img_embeds = vision_outputs.last_hidden_state
    img_embeds = img_embeds[:, 1:]  # remove cls token
    img_embeds_numpy = img_embeds.detach().cpu().numpy()

    # garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    return img_embeds_numpy[0]


@torch.no_grad()
def get_clip_embeddings(image, language="", device="cuda", max_length=77, image_token_size=(3, 3), resize=True):
    """Get CLIP embedding."""
    global global_vision_model, global_vision_processor, global_language_model, global_language_processor
    if global_vision_model is None:
        global_vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        global_vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        global_language_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        global_language_processor = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        print("initialize CLIP Model")

    # language
    if len(language) > max_length:  # one letter and one token
        print(language)
        language = language[:max_length]

    text_inputs = global_language_processor(
        text=[language], return_tensors="pt", padding="max_length", max_length=max_length
    ).to(device)
    text_outputs = global_language_model(**text_inputs)
    text_embeds_numpy = text_outputs.last_hidden_state.detach().cpu().numpy()

    # vision
    image = Image.fromarray(image)
    vision_inputs = global_vision_processor(images=image, return_tensors="pt", do_center_crop=resize).to(device)
    vision_outputs = global_vision_model(**vision_inputs)

    # 16 -> 50 (7*7+1)
    img_embeds = vision_outputs.last_hidden_state
    img_embeds = img_embeds[:, 1:]  # .reshape(1, 7, 7, -1)  # remove cls token
    img_embeds_numpy = img_embeds.detach().cpu().numpy()

    # garbage collection
    del vision_outputs, text_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return text_embeds_numpy[0], img_embeds_numpy[0]


@torch.no_grad()
def get_t5_embeddings(language, per_token=True, max_length=16, device="cpu"):
    """Get T5 embedding"""
    global global_language_model, global_language_processor
    if global_language_model is None:
        global_language_model = T5Model.from_pretrained("t5-base").to(device)
        global_language_processor = T5Tokenizer.from_pretrained("t5-base")

    # forward pass through encoder only
    enc = global_language_processor(
        [language],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)

    output = global_language_model.encoder(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True
    )
    torch.cuda.empty_cache()
    if per_token:
        return output.last_hidden_state[0].detach().cpu().numpy()
    else:
        # get the final hidden states. average across tokens.
        emb = output.last_hidden_state[0].mean(dim=0).detach().cpu().numpy()
        return emb

@torch.no_grad()
def get_vit_embeddings(image, per_token=False, device="cuda", downsample=False):
    """Get VIT embedding. Input: H x W x 3"""
    global global_vision_model

    if global_vision_model is None:
        global_vision_model = vit_base_patch16().to(device)

    image = normalize_image_numpy(image)
    image_th = torch.FloatTensor(image).to(device)[None, None]

    # forward pass through encoder only
    output = global_vision_model.forward_features(image_th)[0]

    if per_token:
        return output.detach().cpu().numpy()
    else:
        # get the final hidden states. average across tokens.
        emb = output.mean(dim=0).detach().cpu().numpy()
        return emb


@torch.no_grad()
def get_resnet_embeddings(image, per_token=False, device="cuda", downsample=False):
    """Get Resnet embedding. Input: H x W x 3"""
    global global_vision_model

    if global_vision_model is None:
        global_vision_model = ResNet().to(device)
    device = global_vision_model.device
    image = normalize_image_numpy(image)
    global_vision_model.eval()
    image_th = torch.FloatTensor(image).to(device)
    if len(image_th.shape) == 3:
        image_th = image_th[None]

    # forward pass through encoder only
    output = global_vision_model.net(image_th)
    if downsample:  # pool to 3 x 3
        output = torch.nn.functional.avg_pool2d(output, 2, 2)

    output = output.reshape(output.shape[0], 512, -1).transpose(1, 2)
    return output.detach().cpu().numpy()


@torch.no_grad()
def get_r3m_embeddings(image, per_token=False, device="cuda"):
    """Get Resnet embedding.
    H x W x 3 -> 1 x D"""
    global global_vision_model
    if global_vision_model is None:
        from r3m import load_r3m
        global_vision_model = load_r3m("resnet18").module  # resnet18, resnet34

    global_vision_model = global_vision_model.to(device)

    # image = normalize_image_numpy(image)
    image = image.transpose(2, 0, 1)
    global_vision_model.eval()
    image_th = torch.FloatTensor(image).to(device)
    if len(image_th.shape) == 3:
        image_th = image_th[None]

    # forward pass through encoder only
    output = global_vision_model(image_th)  # 1 x 512 x 1
    output = output.reshape(output.shape[0], 512, -1).transpose(1, 2)
    return output.detach().cpu().numpy()


@torch.no_grad()
def get_voltron_embeddings(image, per_token=False, device="cuda"):
    """Get Resnet embedding.
    H x W x 3 -> 1 x D"""
    global global_vision_model, global_vision_processor
    if global_vision_model is None:
        # install voltron from https://github.com/siddk/voltron-robotics 
        # Load a frozen Voltron (V-Cond) model & configure a vector extractor
        from voltron import instantiate_extractor, load
        global voltron_vcond, voltron_preprocess
        voltron_vcond, processor = load("v-cond", device=device, freeze=True)
        global_vision_processor = [voltron_vcond, processor]
        global_vision_model = instantiate_extractor(voltron_vcond, n_latents=1)()

    global_vision_model.eval()
    img = global_vision_processor[1]((torch.FloatTensor(image.transpose(2, 0, 1))))[None, ...].to(device)
    visual_features = global_vision_processor[0](img, mode="visual")
    output = global_vision_model(visual_features)
    output = output.reshape(1, 384, -1).transpose(1, 2)
    return output.detach().cpu().numpy()


def update_network_dim(cfg, dataset, policy):
    """
    Update the network dimensions based on the dataset and policy.

    Args:
        cfg (dict): The configuration dictionary.
        dataset (object): The dataset object.
        policy (object): The policy object.

    Returns:
        None
    """
    cfg.stem.state["input_dim"] = dataset.state_dim
    if hasattr(cfg.stem, "image"):  # default multiple copies
        cfg.stem.image["num_of_copy"] = dataset.num_views
    if "prev_actions" in cfg.stem:
        cfg.stem["prev_actions"]["input_dim"] = dataset.one_action_dim
    cfg.head["output_dim"] = dataset.action_dim
    cfg.head["input_dim"] = policy.embed_dim
    cfg.stem.state["output_dim"] = policy.embed_dim
    cfg.stem.image["output_dim"] = policy.embed_dim
    cfg.stem["modality_embed_dim"] = policy.embed_dim
    cfg.network["embed_dim"] = policy.embed_dim


class ActionChunker:
    """
    implement temporal ensemble as in ACT
    """
    def __init__(self,chunk_size, temperature=0.01):
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunked_actions = []

    def update(self, action):
        """
        add action to the chunker. Each item in the list is an action trajectory [T, Da]
        """
        # align all timesteps via shifting by 1 and padding with 0s
        for idx, prev_action in enumerate(self.chunked_actions):
            self.chunked_actions[idx] = np.hstack([prev_action[1:], np.zeros((1,prev_action.shape[1]))])
        self.chunked_actions.append(action)
        if len(self.chunked_actions) > self.chunk_size:
            self.chunked_actions.pop(0)
        
    def __call__(self, curr_action):
        """
        update the chunks with current actions and then run the temporel ensemble
        """
        self.update(curr_action)
        chunked_actions = np.array(self.chunked_actions) # [N, T, Da]
        actions_for_curr_step = chunked_actions[:, 0]
        actions_populated = actions_for_curr_step[actions_for_curr_step.sum(-1) != 0,:]
        exp_weights = np.exp(-self.temperature  * np.arange(len(actions_populated)))[::-1]
        exp_weights = exp_weights[:,None] / exp_weights.sum()
        # [T, 1] x [T, Da]
        raw_action = (actions_populated * exp_weights).sum(axis=0)
        return raw_action

    def reset(self):
        """
        reset the chunker. Each item in the list is an action trajectory [T, Da]
        """
        self.chunked_actions = []

def tabulate_print_state(result_dict):
    """print state dict"""
    result_dict = sorted(result_dict.items())
    headers = ["task", "reward"]
    data = [[kv[0], kv[1]] for kv in result_dict]
    data.append(["avg", np.mean([kv[1] for kv in result_dict])])
    str = tabulate.tabulate(data, headers, tablefmt="psql", floatfmt=".2f")
    print(str)
    return str


def log_results(cfg, total_rewards):
    tabulate_print_state(total_rewards)