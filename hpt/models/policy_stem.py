# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from typing import List
from timm.models.vision_transformer import VisionTransformer

import torch
import torch.nn as nn
from .transformer import CrossAttention, Attention
from einops import rearrange, repeat, reduce


INIT_CONST = 0.02

class PolicyStem(nn.Module):
    """policy stem"""

    def __init__(self, **kwargs):
        super().__init__()

    def init_cross_attn(self, stem_spec, modality: str):
        """ initialize cross attention module and the learnable tokens """
        token_num = getattr(stem_spec.crossattn_latent, modality)
        self.tokens = nn.Parameter(
            torch.randn(1, token_num, stem_spec.modality_embed_dim) * INIT_CONST
        )

        self.cross_attention = CrossAttention(
            stem_spec.modality_embed_dim,
            heads=stem_spec.crossattn_heads,
            dim_head=stem_spec.crossattn_dim_head,
            dropout=stem_spec.crossattn_modality_dropout,
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path : str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the latent representations of input data by attention.

        Args:
            Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.

        Returns:
            Output tensor with latent tokens, shape [32, 16, 128], where 16 is the number
            of tokens and 128 is the dimensionality of each token.

        Examples for vision features from ResNet:
        >>> x = np.random.randn(32, 3, 1, 49, 512)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)

        Examples for proprioceptive features:
        >>> x = np.random.randn(32, 3, 1, 7)
        >>> latent_tokens = model.compute_latent(x)
        >>> print(latent_tokens.shape)
        (32, 16, 128)
        """
        # Initial reshape to adapt to token dimensions
        # (32, 3, 1, 49, 128)
        stem_feat = self(x)  
        stem_feat = stem_feat.reshape(stem_feat.shape[0], -1, stem_feat.shape[-1])  # (32, 147, 128)
        # Replicating tokens for each item in the batch and computing cross-attention
        stem_tokens = self.tokens.repeat(len(stem_feat), 1, 1)  # (32, 16, 128)
        stem_tokens = self.cross_attention(stem_tokens, stem_feat)  # (32, 16, 128)
        return stem_tokens

class STPolicyStem(nn.Module):
    """Policy Stem that tokenizes different modalities into the same latent space. 
    This version implements the spatial temporal version.
    It uses conv2D to handle 1-dimension features [B, T, L, D]
    It uses conv3D to handle 1-dimension features [B, T, H, W, D]
    # https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/model/projector.py
    """
    def __init__(self, dimension=2, **kwargs):
        super().__init__()


    def init(self, stem_spec, modality):
        """ initialize cross attention module and the learnable tokens """
        downsample_tokens = getattr(stem_spec.crossattn_latent, modality)
        stem_modality_spec = getattr(stem_spec, modality)

        if stem_modality_spec.conv_dimension == 2:
            self.conv_dim = stem_modality_spec.conv_dimension
            dim_token = downsample_tokens
            self.conv = nn.Sequential(nn.Conv1d(
                in_channels=stem_modality_spec.input_dim,
                out_channels=stem_modality_spec.output_dim,
                kernel_size=stem_modality_spec.filter_size,
                stride=1,
                padding=stem_modality_shidden_dime_tokens) ** (1./3))
            self.conv = nn.Sequential(nn.Conv3d(
                in_channels=stem_modality_spec.input_dim,
                out_channels=stem_modality_spec.output_dim,
                kernel_size=stem_modality_spec.filter_size,
                stride=1,
                padding=stem_modality_spec.filter_size // 2,
                bias=True
            ), nn.SiLU())
            self.pool = nn.AdaptiveAvgPool3d((dim_token, dim_token, dim_token))

    def compute_latent(self, x):
        """
        Args:
            example x: Input tensor with shape [32, 3, 1, 49, 512] representing the batch size,
            horizon, instance (e.g. num of views), number of features, and feature dimensions respectively.
            Average over the number of instances.
        """
        B, T, I, *_ = x.shape
        x = rearrange(x, 'B T I ... D -> (B I) D T ...')

        if self.conv_dim == 3:
            # assume fixed width and height
            x = rearrange(x, 'B D T (W1 W2) -> B D T W1 W2', 
                          W1=int(x.shape[-1] ** (1/2)),  W2=int(x.shape[-1] ** (1/2)))
        out = self.conv(x)
        out = self.pool(out)
        out = rearrange(out, '(B I) D ... -> B I (...) D', B=B, I=I).mean(dim=1)
        return out


class AttentivePooling(nn.Module):
    """attentive pooling with cross attention"""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, embed_dim) * INIT_CONST)
        self.cross_attention = CrossAttention(embed_dim, heads=8, dim_head=64)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # [B, L, D]
        tokens = self.token.repeat(len(x), 1, 1)
        x = self.cross_attention(tokens, x)
        return x


class MLP(PolicyStem):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
        **kwargs,
    ) -> None:
        """vanilla MLP class"""
        super().__init__()
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList([nn.Sequential(*modules) for _ in range(num_of_copy)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size, 
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]     
        """        
        if self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            y = torch.stack(out, dim=1)
        else:
            y = self.net(x)
        return y


class ResNet(PolicyStem):
    def __init__(
        self,
        output_dim: int = 10,
        weights: str = "DEFAULT",
        resnet_model: str = "resnet18",
        num_of_copy: int = 1,
        **kwargs,
    ) -> None:
        """ResNet Encoder for Images"""  
        super().__init__()
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)

        # by default we use a separate image encoder for each view in downstream evaluation
        self.num_of_copy = num_of_copy
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])

        if num_of_copy > 1:
            self.net = nn.ModuleList(
                [nn.Sequential(*list(pretrained_model.children())[:-2]) for _ in range(num_of_copy)]
            )
        self.input = input
        self.out_dim = output_dim
        self.to_tensor = transforms.ToTensor()
        self.proj = nn.Linear(512, output_dim)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the model.
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size, 
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, 512]     
        """
        B, *_, H, W = x.shape
        x = x.view(len(x), -1, 3, H, W)
        if self.num_of_copy > 1:
            # separate encoding for each view
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            feat = torch.stack(out, dim=1)
        else:
            x = x.view(-1, 3, H, W)
            feat = self.net(x)
        # concat along time
        feat = feat.reshape(B, feat.shape[1], -1).contiguous()
        feat = rearrange(feat, "B L T -> B T L") # (batchsize, number of tokens, resnet feature dimension=512)
        feat = self.proj(feat) # project to (batchsize, number of tokens, output_dim)
        return feat


class PointNet(PolicyStem):
    """Simple Pointnet-Like Network"""

    def __init__(
        self, output_dim: int = 3, input_dim: int = 4, 
        widths: List[int] = [64, 256, 512, 512], dim: int = 1, 
        token_num: int = 64, point_num: int = 1024, **kwargs
    ) -> None:
        super(PointNet, self).__init__()
        input_dim = input_dim[0]
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.point_num = point_num
        self.token_num = token_num

        layers = []
        for oc in widths:
            layers.extend(
                [
                    nn.Conv1d(input_dim, oc, 1, bias=False), 
                    nn.LayerNorm((oc, self.point_num)), 
                ]
            )
            input_dim = oc

        self.linear = nn.Linear(widths[-1], output_dim)
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, T, D, N = inputs.shape
        inputs = inputs.view(-1, D, N)
        x = self.net(inputs)

        token_perpoint_num = self.point_num // self.token_num
        x = x[:, :, ::token_perpoint_num].transpose(-1, -2) 
        token_feat = self.linear(x).view(B, T, -1, self.out_dim)
        return token_feat

def vit_base_patch16(checkpoint_path="output/mae_pretrain_vit_base.pth", **kwargs):
    # load pretrained weights to initialize vit model
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    print("load pretrained model:", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path)["model"], strict=False)
    return model
