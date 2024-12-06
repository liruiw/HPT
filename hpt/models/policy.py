# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
from functools import partial
import hydra
from omegaconf import OmegaConf
from hpt.utils.utils import download_from_huggingface
import os
from typing import List, Optional

import torch
import torch.nn as nn
from ..utils.normalizer import LinearNormalizer
from ..models.transformer import MultiheadAttention, SimpleTransformer
from typing import List
import numpy as np
import einops
from collections import defaultdict

from ..utils.utils import (
    dict_apply,
    get_sinusoid_encoding_table,
    EinOpsRearrange,
    get_image_embeddings,
    normalize_image_numpy,
    get_t5_embeddings,
)

STD_SCALE = 0.02

class Policy(nn.Module):
    """The HPT Policy class.
    Usage for pretraining:
    >>> policy = Policy.from_pretrained("hf://liruiw/hpt-xlarge")
    >>> output = policy.get_action(data_dict)

    Usage for inference:
    >>> policy = Policy.from_pretrained_full_model("hf://liruiw/hpt-xlarge")
    See full usage examples in run.py.

    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_blocks: int = 24,
        num_heads: int = 16,
        use_modality_embedding: bool = True,
        token_postprocessing: bool = False,
        observation_horizon: int = 4,
        action_horizon: int = 1,
        no_trunk: bool = False,
        shared_modality_trunk: Optional[nn.Module] = None,
        use_domain_embedding: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.shared_modality_trunk = shared_modality_trunk
        self.no_trunk = no_trunk

        self.trunk = self._create_policy_trunk(embed_dim, num_blocks, num_heads, **kwargs)
        self.stems = {}
        self.heads = {}
        self.normalizer = {}
        self.encoders = {}
        self.domains = []
        self.use_modality_embedding = use_modality_embedding
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.token_postprocessing = token_postprocessing
        self.modalities_tokens = {}
        self.action_tokens = {}

    def init_encoders(self, modality, encoder):
        """
        Add image/language encoders into the policy parameters in the case of joint finetuning
        """
        self.encoders[modality] = encoder
        self.encoders = nn.ModuleDict(self.encoders)

    def init_domain_stem(self, domain_name, stem_spec):
        """
        Initialize an observation stem for each domain
        """
        self.stem_spec = stem_spec
        self.modalities = stem_spec.modalities
        for modality in self.modalities:
            self.stems[domain_name + "_" + modality] = hydra.utils.instantiate(getattr(stem_spec, modality))
            self.stems[domain_name + "_" + modality].init_cross_attn(stem_spec, modality)
            self.modalities_tokens[modality] = nn.Parameter(torch.randn(1, 1, stem_spec.modality_embed_dim) * STD_SCALE)

        if self.token_postprocessing == "action_token":
            self.action_tokens[domain_name] = nn.Parameter(
                torch.randn(1, self.action_horizon, self.embed_dim) * STD_SCALE
            )

    def init_domain_head(self, domain_name, normalizer=None, head_spec=None):
        """initialize an action head for each domain, along with normalizer"""
        self.head_spec = head_spec
        self.domains.append(domain_name)
        self.heads[domain_name] = hydra.utils.instantiate(head_spec)
        self.normalizer[domain_name] = LinearNormalizer()

        if normalizer is not None:
            self.normalizer[domain_name].load_state_dict(normalizer.state_dict())

    def finalize_modules(self):
        """
        Finalizes the modules of the policy.

        This method converts the stems, heads, normalizer, modalities_tokens
        attentive_pool, and action_tokens into ModuleDict or ParameterDict objects, depending
        on the configuration. It also initializes the weights of the policy.
        """
        self.stems = nn.ModuleDict(self.stems)
        self.heads = nn.ModuleDict(self.heads)
        self.normalizer = nn.ModuleDict(self.normalizer)
        self.modalities_tokens = nn.ParameterDict(self.modalities_tokens)

        self.apply(self._init_weights)
        if self.token_postprocessing == "action_token":
            self.action_tokens = nn.ParameterDict(self.action_tokens)

    def _create_policy_trunk(self, embed_dim: int = 1024, num_blocks: int = 24,
                num_heads: int = 16, drop_path: float = 0.0,
                weight_init_style: str = "pytorch", **kwargs
            ):
        """create the shared representation for pretraining"""

        def instantiate_trunk(embed_dim, num_blocks, num_heads, pre_transformer_ln, add_bias_kv, drop_path):
            return SimpleTransformer(
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                ffn_dropout_rate=0.0,
                drop_path_rate=drop_path,
                attn_target=partial(
                    MultiheadAttention,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=True,
                    add_bias_kv=add_bias_kv,
                ),
                pre_transformer_layer=nn.Sequential(
                    nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else nn.Identity(),
                    EinOpsRearrange("b l d -> l b d"),
                ),
                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
                weight_init_style=weight_init_style,
            )

        trunk = {}
        trunk["trunk"] = instantiate_trunk(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path=drop_path,
        )
        if hasattr(self, "shared_modality_trunk") and self.shared_modality_trunk is not None:
            for modality in self.shared_modality_trunk.modalities:
                trunk[modality] = self.shared_modality_trunk[modality]

        return nn.ModuleDict(trunk)

    def get_position_embedding(self, feature: torch.Tensor, embed_dim: int) -> torch.Tensor:
        """
        Add positional embedding to the features
        """
        tokensize = int(feature.shape[1])
        tokens = get_sinusoid_encoding_table(0, tokensize, self.embed_dim)
        return tokens.repeat((1, 1, 1)).to(feature.device)


    def preprocess_tokens(self, domain: str, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Shared modality layers and add modality tokens. Add positional and time embeddings.
        """
        tokens = torch.cat(features, dim=-2)
        if self.token_postprocessing == "action_token":
            action_tokens = self.action_tokens[domain].repeat(len(tokens), 1, 1)
            tokens = torch.cat([tokens, action_tokens], dim=-2)

        position_tokens = self.get_position_embedding(tokens, self.embed_dim)
        tokens = tokens + position_tokens
        return tokens

    def postprocess_tokens(self, trunk_tokens: torch.Tensor) -> torch.Tensor:
        """
        Postprocesses the trunk tokens to obtain the final features.

        Args:
            trunk_tokens (torch.Tensor): The trunk tokens of shape (N, L, D), where N is the batch size,
                                        L is the sequence length, and D is the token dimension.

        Returns:
            torch.Tensor: The postprocessed tokens of shape (N, D), where N is the batch size and D is the
                          final feature dimension.
        """
        if self.token_postprocessing == "mean":
            return trunk_tokens.mean(dim=1)
        elif self.token_postprocessing == "action_token":
            return trunk_tokens[:, -self.action_horizon :]
        elif self.token_postprocessing == "max":
            return trunk_tokens.max(dim=1)[0]
        elif self.token_postprocessing == "last":
            return trunk_tokens[:, -1]
        elif self.token_postprocessing == "attentive":
            return self.attentive_pool(trunk_tokens)[:, 0]
        elif self.token_postprocessing == "no-op":
            # recommended for transformer decoder
            return trunk_tokens
        else:
            raise ValueError("Invalid token_postprocessing value. Must be one of ['mean', 'action_token', 'max', 'last', 'attentive'].")

    def preprocess_states(self, domain: str, data: dict) -> dict:
        """
        Pre-process proprioception-related inputs, e.g. normalizing states
        """
        if self.stem_spec.normalize_state and "state" in data:
            data["state"] = self.normalizer[domain]["state"].normalize(data["state"])

        if "prev_actions" in data:
            data["prev_actions"] = self.normalizer[domain]["action"].normalize(data["prev_actions"])

        data["state"] = data["state"][:, :, None]
        return data

    def postprocess_actions(self, domain: str, action: torch.Tensor) -> torch.Tensor:
        """
        Postprocess output, e.g. unnormalizing actions
        """
        if domain in self.normalizer:
            action = self.normalizer[domain]["action"].unnormalize(action)
        return action

    def stem_process(self, domain: str, data: dict):
        """
        Pass through the stem to a fixed number of tokens.
        Args:
            data: dictionary of tensors of different modalities
        """
        feats = []
        for modality in self.modalities:
            stem = self.stems[domain + "_" + modality]

            if modality not in data:
                continue

            use_raw_image = "image" in modality and "image" in self.encoders
            if use_raw_image:  # finetuning with encoders
                data[modality] = self.encoders["image"](data[modality])

            # positional embedding for observations
            data_shape = data[modality].shape
            data_horizon = data_shape[1]
            horizon = data_horizon

            if self.train_mode and self.stem_spec.random_horizon_masking and data_horizon > 1:
                horizon = np.random.randint(1, data_horizon + 1)
                data[modality] = data[modality][:, data_horizon - horizon : data_horizon]

            # data is N x T x M x ... x D where M is the # of instances for that sensor
            positional_embedding = get_sinusoid_encoding_table(
                0, horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]).to(data[modality])
            positional_embedding = einops.repeat(positional_embedding, "b h w -> (repeat b) h w", repeat=data_shape[0])
            if not use_raw_image:
                data[modality] = data[modality] + positional_embedding.view(data[modality].shape)
            stem_token = stem.compute_latent(data[modality])
            feats.append(stem_token)

        return feats

    def forward_features(self, domain: str, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the features for the given domain and data.
        Args:
            domain (str): The domain of the data.
            data (Tensor): The input data.
        """
        data = self.preprocess_states(domain, data)

        # stem pass
        self.stem_tokens = self.stem_process(domain, data)

        # combine tokens
        self.trunk_tokens = self.preprocess_tokens(domain, self.stem_tokens)

        # trunk pass
        if not self.no_trunk:
            self.trunk_tokens = self.trunk["trunk"](self.trunk_tokens)

        # pooling the features
        return self.postprocess_tokens(self.trunk_tokens)

    def compute_loss(self, batch):
        """Compute the loss for the training loop forward pass.
        """
        self.train_mode = True
        domain, data = batch["domain"][0], batch["data"]
        features = self.forward_features(domain, data)

        # normalize the labels
        if domain in self.normalizer:
            data["action"] = self.normalizer[domain]["action"].normalize(data["action"])

        # head pass
        loss = self.heads[domain].compute_loss(features, data)
        return loss

    def forward(self, domain: str, data: dict):
        """
        Performs a forward pass of the model.
        Args:
            batch: Dictionary of domain and data, where the data dictionary
            contains the observations (vision, proprioception, etc).
        """
        # pooling the features
        features = self.forward_features(domain, data)

        # head pass
        action = self.heads[domain](features)

        # postprocess. unnormalize the outputs
        action = self.postprocess_actions(domain, action)
        return action

    def print_model_stats(self):
        """Prints out model parameter statistics.

        This method calculates and prints the number of total parameters in the model,
        as well as the number of parameters in each component (stem, trunk, and head).
        It also displays the list of domains associated with the model's heads.
        """
        print("==========================================")
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        head_parameters = sum(p.numel() for p in self.heads.parameters() if p.requires_grad)
        trunk_parameters = sum(p.numel() for p in self.trunk.parameters() if p.requires_grad)
        stem_parameters = sum(p.numel() for p in self.stems.parameters() if p.requires_grad)
        encoder_parameters = (
            sum(p.numel() for p in self.encoders.parameters() if p.requires_grad) if len(self.encoders) > 0 else 0
        )

        print(
            f"number of total params (M): {n_parameters / 1.0e6:.3f} stem: {stem_parameters / 1.0e6:.3f} "
            f"trunk: {trunk_parameters / 1.0e6:.3f} head: {head_parameters / 1.0e6:.3f} encoder: {encoder_parameters / 1.0e6:.3f}"
        )

    def device(self):
        """get the current device of the model"""
        return next(self.parameters()).device

    def save(self, checkpoint_path: str = "./.checkpoints/full"):
        """save the trunk part of the model"""
        try:
            torch.save(self.state_dict(), checkpoint_path)
        except FileNotFoundError:
            logging.warning(f"Could not save module parameters for trunk to {checkpoint_path}.")

    def load_model(self, path: str):
        """load the trunk part of the model"""
        self.load_state_dict(torch.load(path), strict=False)

    def load_trunk(self, path: str):
        """load the trunk part of the model"""
        if "hf://" in path:
            if "output" in path:
                path = path.replace("output/", "")
            path = download_from_huggingface(path[len("hf://") :])
            self.trunk.load_state_dict(torch.load(path), strict=True)

    def freeze_trunk(self, num_layers: int = 0):
        """ freeze the trunk parameters in the last num_layers """
        layers = list(self.trunk["trunk"].children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_trunk(self, num_layers: int = 0):
        """ unfreeze the trunk parameters in the last num_layers  """
        layers = list(self.trunk["trunk"].children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    @classmethod
    def from_pretrained(self, checkpoint_path: str):
        """
        Load a pretrained trunk from the checkpoint and return the policy for transfering.
        """
        checkpoint_path = checkpoint_path.replace("output/", "")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = download_from_huggingface(checkpoint_path[len("hf://") :])
        cfg = OmegaConf.load(os.path.join(checkpoint_path + "/config.yaml"))
        cfg = OmegaConf.structured(cfg)
        cfg.network["_target_"] = "hpt.models.policy.Policy"
        policy = hydra.utils.instantiate(cfg.network)
        policy.load_trunk(os.path.join(checkpoint_path, "trunk.pth"))
        return policy

    @classmethod
    def from_pretrained_full_model(self, checkpoint_path: str, domain: str):
        """
        Load a pretrained full model from a checkpoint and return the policy.
        """
        if not os.path.exists(checkpoint_path):
            checkpoint_path = download_from_huggingface("liruiw/" + checkpoint_path)
        cfg = OmegaConf.load(os.path.join(checkpoint_path + "/config.yaml"))
        cfg = OmegaConf.structured(cfg)
        cfg.network["_target_"] = "hpt.models.policy.Policy"
        policy = hydra.utils.instantiate(cfg.network)

        # here we need to update dimensions without loading the datasets
        policy.init_domain_stem(domain, cfg.stem)
        policy.init_domain_head(domain, None, cfg.head)
        policy.finalize_modules()
        policy.load_model(os.path.join(checkpoint_path, "model.pth"))

        print("stem keys:", policy.stems.keys())
        policy.print_model_stats()

        return policy


    def _init_weights(self, m):
        """
        Weight initialization for transformer
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    ################# Policy Evaluation-Specific #################
    def reset(self):
        """
        Reset the policy's history buffer
        """
        self.history_buffer = defaultdict(list)

        # current steps in open-loop rollouts
        self.openloop_traj_step = self.action_horizon - 1
        self.language_embedding = None

    @torch.no_grad()
    def get_action(self, data: dict, domain: str = None):
        """Get action in the evaluation setup.

        Args:
            data (dict): The input data dictionary.
            domain (str, optional): The domain for which to get the action. Defaults to None.

        Returns:
            torch.Tensor: The action tensor.
        """
        self.train_mode = False

        def update_history_buffer(key: str, new_obs: torch.Tensor):
            """Update the history buffer with a new observation.

            Args:
                key (str): The key for the observation.
                new_obs (torch.Tensor): The new observation tensor.
            """
            # act like a deque
            self.history_buffer[key].append(new_obs)
            if len(self.history_buffer[key]) > self.observation_horizon:
                self.history_buffer[key].pop(0)

        if domain is None:  # default
            domain = self.domains[0]

        if not hasattr(self, "history_buffer"):
            print("should call policy reset explicitly to avoid problems for evaluation in sequence.")
            self.reset()

        action_dim = len(self.normalizer[domain].params_dict["action"]["input_stats"].min)
        device = next(self.parameters()).device
        data_noimg = {k: v for k, v in data.items() if "image" not in k}
        data_img = {k: v for k, v in data.items() if "image" in k}

        # append batch and T dimensions
        data_th = dict_apply(data_noimg, lambda x: torch.FloatTensor(x)[None, None].to(device).float())

        # handle multi-views and history in image
        img_queue = []
        for img_key in data_img.keys():
            if self.stem_spec.precompute_feat and "image" not in self.encoders:
                # precomputed
                img_embedding = get_image_embeddings(data[img_key], self.stem_spec.image_encoder)
                img_queue.append(torch.FloatTensor(img_embedding).to(device).float())
            else:
                # raw inputs
                image = normalize_image_numpy(data[img_key])
                img_queue.append(torch.FloatTensor(image).to(device).float())

        update_history_buffer("image", torch.cat(img_queue, dim=0))  # concat in channel for views
        data_th["image"] = torch.stack(self.history_buffer["image"], dim=0).float()[None]

        # handle state and language
        for modality in data_noimg.keys():
            update_history_buffer(modality, data_th[modality])

            # language is the same for the whole trajectory
            if "language" in modality:
                if "language" in self.modalities:
                    if self.language_embedding is None:
                        self.language_embedding = get_t5_embeddings(data_th[modality], per_token=True)
                    data_th[modality] = self.language_embedding
            else:
                data_th[modality] = torch.cat(self.history_buffer[modality], dim=1).float()

        # handle previous actions
        if "prev_actions" in self.history_buffer:
            data_th["prev_actions"] = torch.cat(self.history_buffer["prev_actions"], dim=1).float()

        if self.openloop_traj_step != self.action_horizon - 1:
            # use previous predictions in open-loop execution
            self.openloop_traj_step += 1
        else:
            action_th = self(domain, data_th)  # forward pass
            self.action_traj = action_th.detach().cpu().numpy()[0]  # batch=1
            self.action_traj = self.action_traj.reshape(-1, action_dim)  # T x Da
            self.openloop_traj_step = 0  # reset steps

        # handle previous actions
        curr_action = self.action_traj[self.openloop_traj_step]
        update_history_buffer("prev_actions", torch.FloatTensor(curr_action)[None, None, None].to(device).float())
        return curr_action
