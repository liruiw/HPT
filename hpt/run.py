# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import hydra
import torch
from torch.utils import data
from tqdm import trange

from hpt.utils import utils
from hpt import train_test
from hpt.utils.warmup_lr_wrapper import WarmupLR
import wandb
from omegaconf import OmegaConf
import numpy as np

MAX_EPOCHS = 100000
TEST_FREQ = 3
MODEL_SAVING_FREQ = 200
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def init_policy(cfg, dataset, domain, device):
    """
    Initialize the policy and load the pretrained model if available.

    Args:
        cfg (Config): The configuration object.
        dataset (Dataset): The dataset object.
        domain (str): The domain of the policy.
        device (str): The device to use for computation.

    Returns:
        Policy: The initialized policy.

    """
    # initialize policy and load pretrained model
    pretrained_exists = len(cfg.train.pretrained_dir) > len("output/") and os.path.exists(
        os.path.join(cfg.train.pretrained_dir, f"trunk.pth")
    )
    if pretrained_exists:
        print("load pretrained trunk config")
        pretrained_cfg = OmegaConf.load(cfg.train.pretrained_dir + "/config.yaml")
        pretrained_cfg = OmegaConf.structured(pretrained_cfg)
        pretrained_cfg.network["_target_"] = "hpt.models.policy.Policy"
        policy = hydra.utils.instantiate(pretrained_cfg.network).to(device)
        print("load trunk from local disk")

    elif "hf" in cfg.train.pretrained_dir:
        from hpt.models.policy import Policy
        policy = Policy.from_pretrained(cfg.train.pretrained_dir)
        print("load trunk from cloud")

    else:
        policy = hydra.utils.instantiate(cfg.network).to(device)
        print("train from scratch!!!")

    utils.update_network_dim(cfg, dataset, policy)
    policy.init_domain_stem(domain, cfg.stem)
    normalizer = dataset.get_normalizer()
    policy.init_domain_head(domain, normalizer, cfg.head)

    # add encoders into policy parameters. enable end-to-end training of the viison model for instance.
    if cfg.network.finetune_encoder:
        utils.get_image_embeddings(np.zeros((320, 240, 3), dtype=np.uint8), cfg.dataset.image_encoder)
        from hpt.utils.utils import global_language_model, global_vision_model
        policy.init_encoders("image", global_vision_model)

    policy.finalize_modules()
    if pretrained_exists:
        policy.load_trunk(os.path.join(cfg.train.pretrained_dir, f"trunk.pth"))

    if cfg.train.freeze_trunk:
        policy.freeze_trunk()
        print("trunk frozen")
    policy.print_model_stats()
    policy.to(device)

    return policy


@hydra.main(config_path="../experiments/configs", config_name="config", version_base="1.2")
def run(cfg):
    """
    This script runs through the trainining loops for downstream task. It loads the pretrained model
    and trains it on the downstream task.
    """
    # initialize run
    date = cfg.output_dir.split("/")[1]
    run = wandb.init(
        project="hpt-transfer",
        tags=[cfg.wb_tag],
        name=f"{date}_{cfg.script_name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=False,
        save_code=False,
        resume="allow",
    )
    utils.set_seed(cfg.seed)
    print(f"train policy models with pretrained: { cfg.train.pretrained_dir}!")
    print("wandb url:", wandb.run.get_url())

    # setup dataset
    device = "cuda"
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    domain = domain_list[0]
    dataset = hydra.utils.instantiate(
        cfg.dataset, dataset_name=domain, env_rollout_fn=cfg.dataset_generator_func, **cfg.dataset
    )
    val_dataset = dataset.get_validation_dataset()
    train_loader = data.DataLoader(dataset, **cfg.dataloader)
    test_loader = data.DataLoader(val_dataset, **cfg.val_dataloader)

    # init policy
    policy = init_policy(cfg, dataset, domain, device)

    # optimizer and scheduler
    opt = utils.get_optimizer(cfg.optimizer, policy, cfg.optimizer_misc)
    cfg.lr_scheduler.T_max = int(cfg.train.total_iters)
    sch = utils.get_scheduler(cfg.lr_scheduler, optimizer=opt)
    sch = WarmupLR(sch, init_lr=0, num_warmup=cfg.warmup_lr.step, warmup_strategy="linear")

    # misc
    utils.save_args_hydra(cfg.output_dir, cfg)
    epoch_size = len(train_loader)
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    cfg.total_num_traj = dataset.replay_buffer.n_episodes
    policy_path = os.path.join(cfg.output_dir, "model.pth")
    print(f"Epoch size: {epoch_size} Traj: {cfg.total_num_traj} Train: {len(dataset)} Test: {len(val_dataset)}")

    # train / test loop
    pbar = trange(MAX_EPOCHS, position=0)
    for epoch in pbar:
        train_stats = train_test.train(cfg.log_interval, policy, device, train_loader, opt, sch, epoch)
        train_steps = (epoch + 1) * len(train_loader)

        if epoch % TEST_FREQ == 0:
            test_loss = train_test.test(policy, device, test_loader, epoch)
            wandb.log({"validate/epoch": epoch, f"validate/{domain}_test_loss": test_loss})

        if "loss" in train_stats:
            print(f"Steps: {train_steps}. Train loss: {train_stats['loss']:.4f}. Test loss: {test_loss:.4f}")

        policy.save(policy_path)
        if train_steps > cfg.train.total_iters:
            break

    # log and finish
    print("model saved to :", policy_path)
    utils.save_args_hydra(cfg.output_dir, cfg)
    pbar.close()
    run.finish()
    wandb.finish()


if __name__ == "__main__":
    run()
