# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.conditional_unet1d import ConditionalUnet1D


class DiffusionPolicy(nn.Module):
    """
    A diffusion-based policy head.

    Args:
        model (ConditionalUnet1D): The model used for prediction.
        noise_scheduler: The noise scheduler used for the diffusion process.
        action_horizon (int): The number of time steps in the action horizon.
        output_dim (int): The dimension of the output.
        num_inference_steps (int, optional): The number of inference steps.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler,
        action_horizon,
        output_dim,
        num_inference_steps=None,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.action_horizon = action_horizon
        self.action_dim = output_dim
        self.kwargs = kwargs
        self.num_inference_steps = num_inference_steps
        if num_inference_steps is None:
            self.num_inference_steps = noise_scheduler.num_train_timesteps

    def conditional_sample(
        self,
        global_cond,
        generator=None,
        **kwargs,
    ):
        """
        Perform conditional sampling using the diffusion process.

        Args:
            global_cond: The global condition.
            generator: The random number generator.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The sampled trajectory.

        """
        model = self.model
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=(len(global_cond), self.action_horizon, self.action_dim),
            dtype=global_cond.dtype,
            device=global_cond.device,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            model_output = model(trajectory, t, global_cond=global_cond)
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator).prev_sample
        return trajectory

    def forward(self, global_cond, **kwargs):
        """
        Perform forward pass through the diffusion-based policy head.

        Args:
            global_cond: The global condition.
            **kwargs: Additional keyword arguments.
        """
        return self.conditional_sample(global_cond)

    def compute_loss(self, global_cond, data):
        """
        Compute the loss for the diffusion-based policy head.

        Args:
            global_cond (torch.Tensor): The global condition tensor of shape (batch_size, global_cond_dim).
            data (dict): The input data dictionary containing the following keys:
                - "action" (torch.Tensor): The action tensor of shape (batch_size, action_horizon * action_dim).

        Returns:
            torch.Tensor: The computed loss.

        """
        trajectory = data["action"].reshape((len(global_cond), self.action_horizon, self.action_dim))  # Reshape the action tensor
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=trajectory.device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)
        target = noise
        return F.mse_loss(pred, target)
