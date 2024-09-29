# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import copy

from hpt.utils.replay_buffer import ReplayBuffer
from hpt.utils.sampler import SequenceSampler, get_val_mask
from hpt.utils.normalizer import LinearNormalizer
from hpt.utils import utils

import time
import os
import zarr
import cv2
import traceback
from collections import defaultdict

def select_image(observation, target_shape=(224, 224), normalize: bool = False, 
                verbose: bool = False, resize: bool = True):
    """
    select a canonical frame as image observation. the order is as follows:
    preferably we use wrist camera view
    otherwise return the first view image.
    """
    imgs = []
    for key in observation:
        if "image" in key:
            image = np.array(observation[key])
            if verbose:
                print("selected image key:", key)
            if resize:
                img = cv2.resize(image, target_shape)
            imgs.append(img)

    return imgs


def select_proprioception(observation, verbose: bool = False):
    """
    select a proprioception representation for the dataset. the order is as follows:
    If there is already a proprioception representation, just use it.
    If there exist end effector and joint positions, add both.
    """
    if "state" in observation:
        state = np.array(observation["state"])
        return state

    joint_and_ee = []
    for key in observation.keys():
        if "joint" in key:
            joint = np.array(observation[key])
            joint_and_ee.append(joint)

    for key in observation.keys():
        if "ee" in key or "pose" in key:
            ee_state = np.array(observation[key])
            joint_and_ee.append(ee_state)

    if len(joint_and_ee) > 0:
        return np.concatenate(joint_and_ee)

    # no propriceptive info is available
    return None

def process_dataset_step(
    dataset_name: str,
    step: dict,
    precompute: bool = False,
    use_multiview: bool = False,
    use_ds: bool = False,
    image_encoder: str = "resnet",
    data_augment_ratio: float = 1.0, 
):
    """map dataset-specific key and values to a unified format"""
    step_dict = {}
    step_dict["action"] = np.array(step["action"])
    state = select_proprioception(step["observation"])
    if state is not None:
        step_dict["state"] = state.astype("float32")

    if precompute:
        # recompute the embeddings. ~0.5s per data
        if "language_instruction" not in step:
            # add dummy if there were no language instructions
            language = ""
        else:
            language = step["language_instruction"]

        images = select_image(step["observation"])

        if image_encoder == "clip":
            # get image and text embeddings together from CLIP.
            image_embeddings = []
            for image in images:
                step_dict["language"], image_embedding = utils.get_clip_embeddings(image, language)
                image_embeddings.append(image_embedding)
                if not use_multiview:
                    break
        else:
            # the default setting is to do separate encoding
            step_dict["language"] = utils.get_t5_embeddings(language, per_token=True, device="cpu")
            image_embeddings = []
            for image in images:
                image_embeddings.append(
                    utils.get_image_embeddings(image, image_encoder, downsample=use_ds, device="cpu")
                )
                if not use_multiview:
                    break

            if len(image_embeddings) > 0:
                step_dict["image"] = np.concatenate(image_embeddings, axis=0)

    else:
        images = [utils.normalize_image_numpy(im) for im in select_image(step["observation"])]
        if len(images) > 0:
            image = np.concatenate(images, axis=0)
            step_dict["image"] = image
        step_dict["language"] = utils.tokenize_language((step["language_instruction"]), per_token=True)

    if "image" in step_dict:
        step_dict["image"] = step_dict["image"].astype("float32")
    if "language" in step_dict:
        step_dict["language"] = step_dict["language"].astype("float32")
    return step_dict


class LocalTrajDataset:
    """
    Single Dataset class that converts simulation data into trajectory data.
    Explanations of parameters are in config
    """

    def __init__(
        self,
        dataset_name: str = "metaworld",
        mode: str = "train",
        episode_cnt: int = 10,
        step_cnt: int = 50000,
        data_augmentation: bool = False,
        data_ratio: int = 1,
        use_disk: bool = False,
        horizon: int = 4,
        pad_before: int = 0,
        pad_after: int = 0,
        val_ratio: float = 0.1,
        seed: int = 233,
        action_horizon: int = 1,
        observation_horizon: int = 1,
        dataset_postfix: str = "",
        dataset_encoder_postfix: str = "",
        precompute_feat: bool = False,
        image_encoder: str = "resnet",
        env_rollout_fn=None,
        use_multiview: bool = False,
        downsample_vision: bool = False,
        normalize_state: bool = False,
        action_multiple_horizon: bool = True,
        regenerate: bool = False,
        data_augment_ratio: int = 1,
        proprioception_expand: bool = False,
        proprioception_expand_dim: int = 1,
        **kwargs,
    ):
        self.dataset_name = dataset_name.strip()
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.data_augmentation = data_augmentation
        self.episode_cnt = episode_cnt
        self.step_cnt = step_cnt
        self.action_horizon = action_horizon
        self.observation_horizon = observation_horizon
        self.horizon = self.action_horizon + self.observation_horizon - 1
        self.action_multiple_horizon = action_multiple_horizon
        self.num_views = 1
        self.data_augment_ratio = data_augment_ratio
        self.proprioception_expand_dim = proprioception_expand_dim
        self.proprioception_expand = proprioception_expand

        self.precompute_feat = precompute_feat
        self.image_encoder = image_encoder
        self.data_ratio = data_ratio
        self.use_multiview = use_multiview
        self.normalize_state = normalize_state
        self.downsample_vision = downsample_vision
        self.dataset_name_withpostfix = self.dataset_name + dataset_encoder_postfix + dataset_postfix

        if use_multiview:
            self.dataset_name_withpostfix = self.dataset_name_withpostfix + "_multiview"
        if downsample_vision:
            self.dataset_name_withpostfix = self.dataset_name_withpostfix + "_ds"
        if data_augment_ratio > 1:
            self.dataset_name_withpostfix = self.dataset_name_withpostfix + f"_aug_{data_augment_ratio}"

        dataset_path = "data/zarr_" + self.dataset_name_withpostfix  # match RTX format
        load_from_cache = os.path.exists(dataset_path + "/data/action") and not regenerate
        print(f"\n\n >>>dataset_path: {dataset_path} load_from_cache: {load_from_cache} \n\n")

        if use_disk or not os.path.exists(dataset_path):  # first time generation
            if load_from_cache:
                self.replay_buffer = ReplayBuffer.create_from_path(dataset_path)
            else:
                os.system(f"rm -rf {dataset_path}")
                self.replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(path=dataset_path))
        else:
            if load_from_cache:
                self.replay_buffer = ReplayBuffer.copy_from_path(dataset_path)
            else:
                self.replay_buffer = ReplayBuffer.create_empty_numpy()

        # loading datasets
        if not load_from_cache:
            self.create_replaybuffer_from_env(env_rollout_fn)

        self.get_training_dataset(val_ratio, seed)
        print("data keys:", self.replay_buffer.data.keys())
        self.get_sa_dim()

    def get_sa_dim(self):
        """
        Get the dimensions of the action and state.

        Returns:
            action_dim (int): The dimension of the action.
            state_dim (int): The dimension of the state.
            num_views (int): The number of views (if images are present in the dataset).
        """
        self.action_dim = self[0]["data"]["action"].shape[-1]
        self.one_action_dim = self[0]["data"]["action"].shape[-1]

        if self.action_multiple_horizon:
            # TxDa For temporal diffusion v.s. (TxDa) for MLP
            self.action_dim *= self.action_horizon
        self.state_dim = self[0]["data"]["state"].shape[-1]
        if "image" in self[0]["data"]:
            self.num_views = self[0]["data"]["image"].shape[0]

    def get_normalizer(self, mode="limits", **kwargs):
        """
        Returns an action normalizer based on the provided mode.

        Returns:
        - normalizer: The action normalizer object.

        """
        data = self._sample_to_data(self.replay_buffer)
        self.normalizer = LinearNormalizer()
        self.normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for k, v in self.normalizer.params_dict.items():
            print(f"normalizer {k} stats min: {v['input_stats'].min}")
            print(f"normalizer {k} stats max: {v['input_stats'].max}")
        return self.normalizer

    def create_replaybuffer_from_env(self, env_rollout_fn):
        """load the dataset from cloud into the disk and create replay buffer"""
        mode = "train"
        max_episode = self.episode_cnt
        prev_traj_end_idx = 0
        traj_len = []

        print(f"Number of samples in {self.dataset_name} {mode} split: {max_episode}")
        start_time = time.time()
        dataset_iter = env_rollout_fn
        episode_cnt = 0

        for traj_idx, episode in enumerate(dataset_iter):
            # load the traj data from the offline trajectory
            if episode is None or len(episode["steps"]) == 0:
                continue

            data = defaultdict(list)
            traj_end_idx = prev_traj_end_idx
            # add episode to replay buffer
            try:
                dataset_steps = []
                for step in episode["steps"]:
                    dataset_step = process_dataset_step(
                        self.dataset_name,
                        step,
                        precompute=self.precompute_feat,
                        image_encoder=self.image_encoder,
                        use_multiview=self.use_multiview,
                        use_ds=self.downsample_vision,
                        data_augment_ratio=self.data_augment_ratio,
                    )
                    dataset_steps.append(dataset_step)

                for dataset_step in dataset_steps:
                    for key, val in dataset_step.items():
                         
                        data[key].append(val)
                    traj_end_idx += 1

                for key, val in data.items():
                    data[key] = np.array(data[key])

                traj_len.append(traj_end_idx - prev_traj_end_idx)
                prev_traj_end_idx = traj_end_idx
                self.replay_buffer.add_episode(data)
                episode_cnt += 1

            except Exception as e:
                print("---------------")
                print("add episode failed:", traj_idx, traceback.format_exc())

            if episode_cnt > max_episode:
                break

            if traj_end_idx > self.step_cnt:
                break

        print(f"Avg {len(traj_len)} traj length: {np.mean(traj_len):.1f} Total: {prev_traj_end_idx}")
        print(f"dataset time: {time.time() - start_time:.3f}")

    def _sample_to_data(self, sample):
        """
        Convert a sample to data format.
        """
        data = {"action": sample["action"]}
        if "state" in sample and self.normalize_state:
            data["state"] = sample["state"]
        return data

    def get_training_dataset(self, val_ratio: float, seed: int):
        """
        Returns the training dataset.

        Args:
            val_ratio (float): The ratio of validation episodes to total episodes.
            seed (int): The random seed for splitting the dataset.
        """
        # split into train and test sets
        self.val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        self.train_mask = ~self.val_mask
        if self.train_mask.sum() == 0:
            self.train_mask = self.val_mask

        # considering hyperparameters and masking
        n_episodes = int(self.data_ratio * min(self.episode_cnt, self.replay_buffer.n_episodes))
        self.val_mask[n_episodes:] = False
        self.train_mask[n_episodes:] = False

        # normalize and create sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.train_mask,
        )
        print(
            f"{self.dataset_name} size: {len(self.sampler)} episodes: {n_episodes} train: {self.train_mask.sum()} eval: {self.val_mask.sum()}"
        )

    def get_validation_dataset(self):
        """
        Returns a validation dataset.

        This method creates a copy of the current dataset and sets the necessary attributes
        to create a validation dataset. It sets the sampler to a SequenceSampler with the
        appropriate parameters, sets the train_mask to the val_mask, and returns the new
        validation dataset.

        Returns:
            val_set (LocalTrajDataset): A validation dataset.

        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.train_mask = self.val_mask
        return val_set

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int):
        """get data item for each trajectory sequence"""
        sample = self.sampler.sample_sequence(idx)
        for key, val in sample.items():
            if key != "action":
                if self.proprioception_expand and key == "state":
                    sample[key] = np.tile(sample[key][..., None], (1, 1, self.proprioception_expand_dim))                
                if key == "language":
                    sample[key] = val[:1]
                else:
                    sample[key] = val[: self.observation_horizon]

            else:
                # future actions
                sample["action"] = val[
                    self.observation_horizon - 1 : self.action_horizon + self.observation_horizon - 1
                ]
        return {"domain": self.dataset_name, "data": sample}

if __name__ == '__main__':
    import torch
    import tqdm
    
    dataset = LocalTrajDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    for batch in tqdm.tqdm(dataloader):
        pass    