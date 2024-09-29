# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
 
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict

RESOLUTION = (480, 480)

# define your own dataset conversion
def convert_dataset_image(dataset_dir="realworld_data", env_names=None, gui=False, episode_num_pertask=5000):
    # convert to a list of episodes that can be added to replay buffer
    for env_name in env_names:
        data = dict(np.load(f"data/realworld_data/{env_name}/demo_state.npz", allow_pickle=True))
        traj_end_indexes = np.where(data["done"])[0]
        prev_traj_end_idx = 0

        for traj_idx, traj_end_idx in tqdm(enumerate(traj_end_indexes[:episode_num_pertask])):

            action = data["action"][prev_traj_end_idx:traj_end_idx].astype(np.float32)  # remove fingers
            image = data["image"][prev_traj_end_idx:traj_end_idx, ..., :3].astype(np.float32)
            image2 = data["image2"][prev_traj_end_idx:traj_end_idx, ..., :3].astype(np.float32)
            ee_pose = data["ee_pose"][prev_traj_end_idx:traj_end_idx].astype(np.float32)
            state = ee_pose

            # append the action
            lang = data["task_name"][0]
            steps = []
            for a, o1, o2, s in zip(action, image, image2, state):
                # break into step dict
                step = {
                    "observation": {"image": o1, "image2": o2, "state": s},
                    "action": a,
                    "language_instruction": lang,
                }
                steps.append(OrderedDict(step))

            data_dict = {"steps": steps}
            prev_traj_end_idx = traj_end_idx
            yield data_dict


class RolloutRunner:
    """evaluate policy rollouts"""

    def __init__(self, env_names, episode_num, save_video=False):
        pass

    @torch.no_grad()
    def run(self, policy, save_video=False, gui=False, video_postfix="", seed=233, env_name=None, **kwargs):
        return 0, 0
