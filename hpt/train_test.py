# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import time

import numpy as np
import torch
from tqdm import tqdm
import hydra
import numpy as np

from collections import deque
from hpt.utils.utils import dict_apply
from hpt.utils.model_utils import module_max_gradient, module_mean_param
import wandb

info_key = [
    "loss",
    "mae_loss",
    "max_action",
    "max_label_action",
    "max_gradient",
    "max_stem_gradient",
    "max_trunk_gradient",
    "max_head_gradient",
    "mean_stem_param",
    "mean_trunk_param",
    "mean_head_param",
    "step_time",
    "data_time",
    "epoch",
    "mean_param",
    "lr",
]

info_log = {k: deque([], maxlen=50) for k in info_key}

def log_stat(info_log, train_step, log_interval, log_name, domain, loss, model, optimizer, step_time, data_time, epoch):
    """
    log wandb statistics for training
    """
    if domain + "_loss" not in info_log:
        info_log[domain + "_loss"] = deque([], maxlen=50)
    info_log[domain + "_loss"].append(loss.item())
    info_log["loss"].append(loss.item())
    info_log["max_gradient"].append(module_max_gradient(model))
    info_log["max_stem_gradient"].append(module_max_gradient(model.stems))
    info_log["max_trunk_gradient"].append(module_max_gradient(model.trunk))
    info_log["max_head_gradient"].append(module_max_gradient(model.heads))
    info_log["mean_stem_param"].append(module_mean_param(model.stems))
    info_log["mean_trunk_param"].append(module_mean_param(model.trunk))
    info_log["mean_head_param"].append(module_mean_param(model.heads))
    info_log["mean_param"].append(module_mean_param(model))
    info_log["step_time"].append(step_time)
    info_log["data_time"].append(data_time)
    info_log["epoch"].append(epoch)
    info_log["lr"].append(optimizer.param_groups[0]["lr"])
    wandb_metrics = {f"{log_name}/{k}": np.mean(v) for k, v in info_log.items() if len(v) > 0}
    wandb.log({"train_step": train_step, **wandb_metrics})


def train(
    log_interval, model, device, train_loader, optimizer, scheduler, epoch, log_name="train",
):
    """
    Training function for one epoch on the train_loader.

    Args:
        log_interval (int): The interval for logging training progress.
        model: The model to be trained.
        device: The device to be used for training.
        train_loader: The data loader for training data.
        optimizer: The optimizer used for updating model parameters.
        scheduler: The learning rate scheduler.
        epoch (int): The current epoch number.
        log_name (str, optional): The name used for logging. Defaults to "train".

    Returns:
        dict: A dictionary containing the average values of logged statistics.
    """
    model.train()
    start_time = time.time()

    epoch_size = len(train_loader)
    pbar = tqdm(train_loader, position=1, leave=True)

    # randomly sample a dataloader with inverse probability square root to the number of data
    for batch_idx, batch in enumerate(pbar):
        batch["data"] = dict_apply(batch["data"], lambda x: x.to(device, non_blocking=True).float())
        data_time = time.time() - start_time
        start_time = time.time()
        domain_loss = model.compute_loss(batch)
        optimizer.zero_grad()
        domain_loss.backward()

        optimizer.step()
        scheduler.step()

        train_step = len(train_loader) * epoch + batch_idx
        step_time = time.time() - start_time
        start_time = time.time()
        log_stat(info_log, train_step, log_interval, log_name, batch["domain"][0],
                domain_loss, model, optimizer, step_time, data_time, epoch)

        pbar.set_description(
            f"Epoch: {epoch} {train_step} Step: {batch_idx}/{epoch_size} Time: {step_time:.3f}"
            f"{data_time:.3f} Loss: {info_log[batch['domain'][0] + '_loss'][-1]:.3f} Grad: {info_log['max_gradient'][-1]:.3f}"
        )

    return {k: np.mean(v) for k, v in info_log.items() if len(v) > 1}


@torch.no_grad()
def test(model, device, test_loader, epoch):
    """
    Evaluate imitation losses on the test sets.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device to run the evaluation on.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
        epoch (int): The current epoch number.

    Returns:
        float: The average test loss.
    """
    model.eval()
    test_loss, num_examples = 0, 0
    pbar = tqdm(test_loader, position=2, leave=False)

    for batch_idx, batch in enumerate(pbar):
        batch["data"] = dict_apply(batch["data"], lambda x: x.to(device, non_blocking=True).float())
        loss = model.compute_loss(batch)

        # logging
        test_loss += loss.item()
        num_examples += 1
        pbar.set_description(
            f"Test Epoch: {epoch} Step: {batch_idx} Domain: {batch['domain'][0]} Loss: {test_loss / (num_examples + 1):.3f}"
        )
    return test_loss / (num_examples + 1)


def eval_policy(policy, cfg, env_name=None, eps_num=-1):
    """
    Evaluate the given policy on the specified environment.

    Args:
        policy: The policy to evaluate.
        cfg: The configuration object.
        env_name: The name of the environment to evaluate on. (default: None)
        eps_num: The episode number. (default: -1)

    Returns:
        success: A boolean indicating whether the evaluation was successful.
        reward: The total reward obtained during the evaluation.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    policy.eval()

    rollout_runner = hydra.utils.instantiate(cfg.rollout_runner)
    rollout_runner.save_video = rollout_runner.save_video and cfg.seed == 0
    video_postfix = cfg.output_dir.split("/")[-2]
    success, reward = rollout_runner.run(policy=policy, env_name=env_name, video_postfix=video_postfix)

    del rollout_runner
    return success, reward


def eval_policy_sequential(policy, cfg, eps_num=-1):
    """
    Evaluate the policy using rollout_runner for each environment sequentially.

    Args:
        policy: The policy to evaluate.
        cfg: The configuration object.
        eps_num (optional): The number of episodes to evaluate. Default is -1.

    Returns:
        A dictionary containing the total rewards for each environment.
    """
    env_names = [env_name.strip() for env_name in cfg.env_names]
    if eps_num > 0:  # cross validate
        env_names = [env_names[0]]

    total_success_list = []

    for env_name in zip(env_names):
        success, rewards = eval_policy(policy, cfg, env_name, eps_num)
        total_success_list.append(success)

    total_rewards = {env_name: rew for env_name, rew in zip(env_names, total_success_list)}
    return total_rewards
