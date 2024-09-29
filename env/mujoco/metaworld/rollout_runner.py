# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import functools

import cv2
import numpy as np
from .envs.mujoco.sawyer_xyz.test_scripted_policies import (
    ALL_ENVS,
    test_cases_latest_nonoise,
)

import torch
from collections import OrderedDict
from tqdm import tqdm
import traceback

ALL_TASK_CONFIG = [
    # env, action noise pct, cycles, quit on success
    ("assembly-v2", np.zeros(4), 10000, True, "pick up the pole and put the circle through the cylinder."),
    ("basketball-v2", np.zeros(4), 10000, True, "pick up the basketball and drop it through the basket."),
    ("bin-picking-v2", np.zeros(4), 10000, True, "move the block from one bin to another bin."),
    ("box-close-v2", np.zeros(4), 10000, True, "close the lid of the box."),
    ("button-press-topdown-v2", np.zeros(4), 10000, True, "press down the button."),
    ("button-press-topdown-wall-v2", np.zeros(4), 10000, True, "press down the button with one finger."),
    ("button-press-v2", np.zeros(4), 10000, True, "press in the button with one finger."),
    ("button-press-wall-v2", np.zeros(4), 100, True, "move between wall and botton and open fingers."),
    ("coffee-button-v2", np.zeros(4), 10000, True, "touch the coffee mug."),
    ("coffee-pull-v2", np.zeros(4), 10000, True, "grasp the coffee mug."),
    ("coffee-push-v2", np.zeros(4), 10000, True, "push the coffee mug."),
    ("dial-turn-v2", np.zeros(4), 10000, True, "turn the dial with the fingers."),
    ("disassemble-v2", np.zeros(4), 10000, True, "pull the circle bar out of the cylinder."),
    ("door-close-v2", np.zeros(4), 10000, True, "close the door with the fingers."),
    ("door-lock-v2", np.zeros(4), 10000, True, "lock the door."),
    ("door-open-v2", np.zeros(4), 10000, True, "open the door."),
    ("door-unlock-v2", np.zeros(4), 10000, True, "unlock the door."),
    ("hand-insert-v2", np.zeros(4), 10000, True, "pick up the wooden block and put it in the box."),
    ("drawer-close-v2", np.zeros(4), 10000, True, "close the drawer with the fingers."),
    ("drawer-open-v2", np.zeros(4), 10000, True, "open the drawer with the fingers."),
    ("faucet-open-v2", np.zeros(4), 10000, True, "turn the faucet to open."),
    ("faucet-close-v2", np.zeros(4), 10000, True, "turn the faucet to close."),
    ("hammer-v2", np.zeros(4), 10000, True, "grasp the hammer and move towards the button."),
    ("handle-press-side-v2", np.zeros(4), 10000, True, "press the handle down."),
    ("handle-press-v2", np.zeros(4), 10000, True, "use the side finger to press the handle."),
    ("handle-pull-side-v2", np.zeros(4), 10000, True, "pull the handle"),
    ("handle-pull-v2", np.zeros(4), 10000, True, "pull the handle."),
    ("lever-pull-v2", np.zeros(4), 10000, True, "pull the lever."),
    ("pick-place-wall-v2", np.zeros(4), 10000, True, "pick the red cylinder and place it at the blue spot."),
    ("pick-out-of-hole-v2", np.zeros(4), 10000, True, "pick up the red cylinder."),
    ("push-back-v2", np.zeros(4), 10000, True, "move the wooded brick to the green spot."),
    ("push-v2", np.zeros(4), 10000, True, "push the red cylinder to the green spot."),
    ("pick-place-v2", np.zeros(4), 10000, True, "pick up the red cylinder and put it in the blue spot."),
    ("plate-slide-v2", np.zeros(4), 10000, True, "pick up the gray cylinder and move it to the red bucket."),
    ("plate-slide-side-v2", np.zeros(4), 10000, True, "push the gray cylinder into the red bucket"),
    ("plate-slide-back-v2", np.zeros(4), 10000, True, "move the gray cylinder out of the red bucket"),
    ("plate-slide-back-side-v2", np.zeros(4), 10000, True, "pull the gray cylinder out of the red bucket"),
    ("peg-insert-side-v2", np.zeros(4), 10000, True, "insert the green peg into the red wall."),
    ("peg-unplug-side-v2", np.zeros(4), 10000, True, "unplug the gray cylinder out."),
    ("soccer-v2", np.zeros(4), 10000, True, "push the soccer ball into the goal net."),
    ("stick-push-v2", np.zeros(4), 10000, True, "pick up the blue box and push the gray."),
    ("stick-pull-v2", np.zeros(4), 10000, True, "pick the blue box and pull the gray."),
    ("push-wall-v2", np.zeros(4), 10000, True, "move the red cylinder around the wall to the green spot."),
    ("reach-wall-v2", np.zeros(4), 10000, True, "reach the red spot on the wall."),
    ("reach-v2", np.zeros(4), 10000, True, "reach the wall."),
    ("shelf-place-v2", np.zeros(4), 10000, True, "put the blue block inside the shelf."),
    ("sweep-into-v2", np.zeros(4), 10000, True, "pick up the wooden block and put inside the hole."),
    ("sweep-v2", np.zeros(4), 10000, True, "pick the wooden block and then drop it."),
    ("window-open-v2", np.zeros(4), 10000, True, "open the window by pushing the handle."),
    ("window-close-v2", np.zeros(4), 10000, True, "cloe the window by pulling the handle."),
]
RESOLUTION = (128, 128)


def writer_for(tag, fps, res, src_folder="demonstrations"):
    if not os.path.exists(src_folder):
        os.mkdir(src_folder)
    return cv2.VideoWriter(
        f"{src_folder}/{tag}.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        res,
    )


@torch.no_grad()
def learner_trajectory_generator(env, policy, lang="", camera_name="view_1"):
    """generate a trajectory rollout from the policy and a metaworld environment"""
    env.reset()
    env.reset_model()
    policy.reset()
    o = env.reset()
    img = env.sim.render(*RESOLUTION, mode="offscreen", camera_name=camera_name)[:, :, ::-1].copy()

    def get_observation_dict(o, img):
        step_data = {"state": o, "image": img, "language_instruction": lang}  # [:3]only use the end effector positions
        return OrderedDict(step_data)

    step_data = get_observation_dict(o, img)

    for _ in range(env.max_path_length):
        a = policy.get_action(step_data)
        o, r, done, info = env.step(a)

        img = env.sim.render(*RESOLUTION, mode="offscreen", camera_name=camera_name)[:, :, ::-1]
        img = cv2.resize(img, RESOLUTION).astype(np.uint8)
        ret = [o, r, done, info, img]
        step_data = get_observation_dict(o, img)
        yield ret


class RolloutRunner:
    """evaluate policy rollouts"""

    def __init__(self, env_names, episode_num, save_video=False):
        self.env_names = env_names
        self.episode_num = episode_num
        self.save_video = save_video

    @torch.no_grad()
    def run(
        self,
        policy,
        save_video=False,
        gui=False,
        video_postfix="",
        video_path=None,
        env_name=None,
        seed=233,
        episode_num=-1,
        **kwargs,
    ):
        """
        Run the rollout for the given policy.

        Args:
            policy: The policy to use for the rollout.
            save_video (bool, optional): Whether to save the video of the rollout. Defaults to False.
            gui (bool, optional): Whether to display the GUI during the rollout. Defaults to False.
            video_postfix (str, optional): The postfix to add to the video filename. Defaults to "".
            video_path (str, optional): The path to save the video. Defaults to None.
            env_name (str, optional): The name of the environment to run the rollout on. Defaults to None.
            seed (int, optional): The seed for the environment. Defaults to 233.
            episode_num (int, optional): The number of episodes to run. Defaults to -1.
            **kwargs: Additional keyword arguments.

        Returns:
            float: The average success rate over the episodes.
            float: The average reward over the episodes.
        """
        camera = "view_1"  # has been modified to wrist view
        flip = True
        noise = np.zeros(4)
        quit_on_success = True
        if episode_num == -1:
            episode_num = self.episode_num  # upper bound for number of trajectories
        all_env_names = [task for (task, _, _, _, lang) in ALL_TASK_CONFIG]
        env_lang_map = {task[0]: lang for (task, _, _, _, lang) in ALL_TASK_CONFIG}

        if type(self.env_names) is not list:  # test on all
            env_names = all_env_names
        else:
            env_names = self.env_names.split(",")

        for env in env_names:
            env = env.strip()
            if env not in all_env_names:
                continue

            language_instruction = [task for (task, _, _, _, lang) in ALL_TASK_CONFIG if task == env][0]
            if env_name is not None:
                env_name = env_name
                if str(env_name[0]) != str(env):
                    continue

            print("env_name:", env_name, episode_num)
            tag = env
            env_keys = sorted(list(ALL_ENVS.keys()))
            env = ALL_ENVS[env]()
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.seed(seed)

            if self.save_video:
                writer = writer_for(
                    tag + f"_{video_postfix}",
                    env.metadata["video.frames_per_second"],
                    RESOLUTION,
                    src_folder="output/output_figures/output_videos/metaworld",
                )

            total_success = 0
            total_reward = 0
            pbar = tqdm(range(episode_num), position=1, leave=True)
            try:
                for i in pbar:
                    eps_reward = 0
                    traj_length = 0
                    q_pos = []

                    step = 0
                    for o, r, done, info, img in learner_trajectory_generator(env, policy, language_instruction):
                        traj_length += 1
                        eps_reward += r
                        if self.save_video and i <= 5:
                            if gui:
                                cv2.imshow("img", img)
                                cv2.waitKey(1)
                            writer.write(img)

                        if info["success"]:
                            break

                        step += 1
                    pbar.set_description(f"success: {info['success']}")
                    total_success += info["success"]
                    total_reward += eps_reward
            except Exception as e:
                print(traceback.format_exc())
            return total_success / episode_num, total_reward / episode_num


@torch.no_grad()
def expert_trajectory_generator(env, policy, camera_name="view_1"):
    """generate a trajectory rollout from the policy and a metaworld environment"""
    env.reset()
    env.reset_model()
    o = env.reset()
    for _ in range(env.max_path_length):
        a = policy.get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, done, info = env.step(a)
        # only use the end effector positions

        img = env.sim.render(*RESOLUTION, mode="offscreen", camera_name=camera_name)[:, :, ::-1]
        img = cv2.resize(img, RESOLUTION).astype(np.uint8)
        ret = [a, o, r, done, info, img]
        yield ret


def generate_dataset_rollouts(
    env_names, save_video=False, gui=False, max_total_transition=2000, episode_num_pertask=100, **kwargs
):
    """online generate scripted expert data for a env"""
    # hyperparameters
    camera = "view_1"  # has been modified to wrist view
    flip = True
    noise = np.zeros(4)
    quit_on_success = True
    cycles = episode_num_pertask // len(env_names)  # upper bound for number of trajectories
    all_env_names = [task for (task, _, _, _, lang) in ALL_TASK_CONFIG]

    if env_names == "all":
        env_names = all_env_names
    else:
        env_names = list(env_names)

    print("metaworld env names:", env_names)
    for env in env_names:
        env = env.strip()
        if env not in all_env_names:
            continue

        language_instruction = [task for (task, _, _, _, lang) in ALL_TASK_CONFIG if task == env][0]
        tag = env
        policy = functools.reduce(lambda a, b: a if a[0] == env else b, test_cases_latest_nonoise)[1]
        env_keys = sorted(list(ALL_ENVS.keys()))
        env = ALL_ENVS[env]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True

        if save_video:
            writer = writer_for(tag, env.metadata["video.frames_per_second"], RESOLUTION)

        dataset_traj_states = []
        dataset_traj_actions = []
        dataset_traj_images = []
        total_transition_num = 0

        for i in range(cycles):
            eps_reward = 0
            traj_length = 0
            eps_states = []
            eps_actions = []
            eps_images = []
            q_pos = []

            step = 0
            try:
                for a, o, r, done, info, img in expert_trajectory_generator(env, policy):
                    eps_states.append(o)
                    eps_actions.append(a)
                    eps_images.append(img)
                    traj_length += 1
                    eps_reward += info["success"]
                    if save_video and i <= 10:
                        if gui:
                            cv2.imshow("img", img)
                            cv2.waitKey(1)
                        writer.write(img)

                    if info["success"]:
                        break

                    step += 1
            except:
                print(traceback.format_exc())
                continue

            print("success:", info["success"])
            if info["success"]:
                # only add successeful examples
                total_transition_num += len(eps_images)
            else:
                continue

            print(f"data generation number of episodes: {tag} {i} {total_transition_num}")
            steps = []

            # the action is associated with the previous state and image
            eps_actions = eps_actions[1:]
            for state, action, image in zip(eps_states, eps_actions, eps_images):
                step = {
                    "action": action,
                    "observation": {"state": state, "image": image},
                    "language_instruction": language_instruction,
                }
                steps.append(step)
            data_dict = {"steps": steps}
            yield data_dict


if __name__ == "__main__":
    # generate for all tasks
    runner = RolloutRunner(["all"], 200)
