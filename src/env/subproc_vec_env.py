import warnings
from typing import Callable, List, Optional, Union, Sequence, Dict

import gymnasium as gym
import numpy as np
from numpy import ndarray
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.base_vec_env import tile_images, VecEnvStepReturn


class CustomSubprocVecEnv(SubprocVecEnv):

    def __init__(self,
                 env_fns: List[Callable[[], gym.Env]],
                 start_method: Optional[str] = None):
        super().__init__(env_fns, start_method)
        self.can_see_walls = True
        self.image_noise_scale = 0.0
        self.image_rng = None  # to be initialized with run id in ppo_rollout.py

    def send_reset(self, env_id: int) -> None:
        # SB3 worker expects a tuple (seed, options) for the "reset" command.
        # Sending (None, None) for a default reset.
        self.remotes[env_id].send(("reset", (None, None)))

    def invisibilize_obstacles(self, obs: np.ndarray) -> np.ndarray:
        # Algorithm A5 in the Technical Appendix
        # For MiniGrid envs only
        obs = np.copy(obs)
        for r in range(len(obs[0])):
            for c in range(len(obs[0][r])):
                # The color of Walls is grey
                # See https://github.com/Farama-Foundation/gym-minigrid/blob/20384cfa59d7edb058e8dbd02e1e107afd1e245d/gym_minigrid/minigrid.py#L215-L223
                # COLOR_TO_IDX['grey']: 5
                if obs[1][r][c] == 5 and 0 <= obs[0][r][c] <= 2:
                    obs[1][r][c] = 0
                # OBJECT_TO_IDX[0,1,2]: 'unseen', 'empty', 'wall'
                if 0 <= obs[0][r][c] <= 2:
                    obs[0][r][c] = 0
        return obs

    def add_noise(self, obs: np.ndarray) -> np.ndarray:
        # Algorithm A4 in the Technical Appendix
        # Add noise to observations
        obs = obs.astype(np.float64)
        obs_noise = self.image_rng.normal(loc=0.0, scale=self.image_noise_scale, size=obs.shape)
        return obs + obs_noise

    def recv_obs(self, env_id: int) -> ndarray:
        # Worker sends (observation, reset_info) upon "reset" command.
        obs, _ = self.remotes[env_id].recv()  # Unpack the tuple
        obs = VecTransposeImage.transpose_image(obs)
        if not self.can_see_walls:
            obs = self.invisibilize_obstacles(obs)
        if self.image_noise_scale > 0:
            obs = self.add_noise(obs)
        return obs

    def step_wait(self) -> VecEnvStepReturn:
        # Call the parent's step_wait, which handles:
        # - receiving results: obs_list, rews, dones, infos, self.reset_infos
        # - stacking observations using _stack_obs
        # - stacking rewards and dones
        # It returns: stacked_obs, stacked_rews, stacked_dones, infos_list
        obs_arr_stacked, rews_stacked, dones_stacked, infos_list = super().step_wait()

        # Apply custom modifications to the stacked observations.
        # Create a writable copy to ensure modifications don't affect unexpected parts.
        processed_obs_arr = np.copy(obs_arr_stacked)

        # Assuming obs_arr_stacked (and thus processed_obs_arr) is a NumPy array
        # where each processed_obs_arr[idx] is a full observation for one environment.
        # If the observation space is a Dict, this loop might need adjustment
        # to target specific keys within each processed_obs_arr[idx].
        # The custom methods invisibilize_obstacles and add_noise appear to expect
        # a single ndarray per observation.
        for idx in range(self.num_envs):
            if not self.can_see_walls:
                processed_obs_arr[idx] = self.invisibilize_obstacles(processed_obs_arr[idx])
            if self.image_noise_scale > 0:
                # add_noise internally casts its input to float64
                processed_obs_arr[idx] = self.add_noise(processed_obs_arr[idx])
        
        return processed_obs_arr, rews_stacked, dones_stacked, infos_list

    def get_first_image(self) -> Sequence[np.ndarray]:
        for pipe in self.remotes[:1]:
            # gather images from subprocesses
            # The worker's "render" command does not use the data payload.
            # It calls env.render(). Ensure env is set to "rgb_array" mode.
            pipe.send(("render", None)) # Sending "rgb_array" as data is ignored by SB3 worker
        imgs = [pipe.recv() for pipe in self.remotes[:1]]
        return imgs

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        # This custom render method only renders the first environment.
        # SB3's default render method in VecEnv renders and tiles all environments.
        try:
            imgs = self.get_first_image()
        except NotImplementedError:
            warnings.warn(f"Render not defined for {self}")
            return None

        # Create a big image by tiling images from the first subprocess
        bigimg = tile_images(imgs) # imgs will have only one image
        if mode == "human":
            import cv2  # pytype:disable=import-error
            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by this custom VecEnv render method")