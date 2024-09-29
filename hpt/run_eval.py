# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import hydra

from hpt.utils import utils
from hpt import train_test
import numpy as np


@hydra.main(config_path="../experiments/configs", config_name="config", version_base="1.2")
def run_eval(cfg):
    """
    This script runs through the eval loop. It loads the model and run throug hthe rollout functions.
    """
    cfg.output_dir = cfg.output_dir + "/" + str(cfg.seed)
    utils.set_seed(cfg.seed)
    print(cfg)

    device = "cuda"
    domain_list = [d.strip() for d in cfg.domains.split(",")]
    domain = domain_list[0]

    # initialize policy
    policy = hydra.utils.instantiate(cfg.network).to(device)
    policy.init_domain_stem(domain, cfg.stem)
    policy.init_domain_head(domain, None, cfg.head)
    policy.finalize_modules()
    policy.print_model_stats()
    utils.set_seed(cfg.seed)

    # add encoders into policy parameters
    if cfg.network.finetune_encoder:
        utils.get_image_embeddings(np.zeros((320, 240, 3), dtype=np.uint8), cfg.dataset.image_encoder)
        from hpt.utils.utils import global_vision_model
        policy.init_encoders("image", global_vision_model)

    # load the full model
    policy.load_model(os.path.join(cfg.train.pretrained_dir, "model.pth"))
    policy.to(device)
    policy.eval()
    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

    # evaluate jointly trained policy
    total_rewards = train_test.eval_policy_sequential(policy, cfg)

    # save the results
    utils.log_results(cfg, total_rewards)


if __name__ == "__main__":
    run_eval()
