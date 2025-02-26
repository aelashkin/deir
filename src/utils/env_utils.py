import os
import warnings
from typing import Any, Callable, Dict, Optional, Type, Union, Sequence
import multiprocessing as mp

import envpool
import gymnasium as gym
import numpy as np

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecEnvWrapper, VecMonitor
from envpool.python.protocol import EnvPool
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, tile_images

# From Stable Baseline 3
# https://github.com/DLR-RM/stable-baselines3/blob/18f4e3ace084a2fd3e0a3126613718945cf3e5b5/stable_baselines3/common/env_util.py

from packaging import version
is_legacy_gym = version.parse(gym.__version__) < version.parse("0.26.0")


class EnvPoolVecAdapter(VecEnvWrapper):
    """
    Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.
    :param venv: The envpool object.
    """

    def __init__(self, venv: EnvPool):
        # Retrieve the number of environments from the config
        venv.num_envs = venv.spec.config.num_envs
        super().__init__(venv=venv)
        self.venv.obs = None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def reset(self) -> VecEnvObs:
        if is_legacy_gym:
            obs = self.venv.reset()
        else:
            obs = self.venv.reset()[0]
        self.venv.obs = obs
        return obs

    def seed(self, seed: Optional[int] = None) -> None:
        # You can only seed EnvPool env by calling envpool.make()
        pass

    def step_wait(self) -> VecEnvStepReturn:
        if is_legacy_gym:
            obs, rewards, dones, info_dict = self.venv.step(self.actions)
        else:
            obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
            dones = terms + truncs

        infos = []
        # Convert dict to list of dict
        # and add terminal observation
        for i in range(self.num_envs):
            infos.append(
                {
                    key: info_dict[key][i]
                    for key in info_dict.keys()
                    if isinstance(info_dict[key], np.ndarray)
                }
            )
            if dones[i]:
                infos[i]["terminal_observation"] = obs[i]
                if is_legacy_gym:
                    obs[i] = self.venv.reset(np.array([i]))
                else:
                    obs[i] = self.venv.reset(np.array([i]))[0]
        self.venv.obs = obs
        return obs, rewards, dones, infos

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if self.venv.obs is None:
            return

        try:
            imgs = self.venv.obs
        except NotImplementedError:
            warnings.warn(f"Render not defined for {self}")
            return

        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs[:1])

        bigimg_size = bigimg.shape[-1]
        bigimg = bigimg[-1].reshape(bigimg_size, bigimg_size)

        # grayscale to fake-RGB
        bigimg = np.stack((bigimg,) * 3, axis=-1)

        if mode == "human":
            import cv2  # pytype:disable=import-error
            cv2.imshow("vecenv", bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported by VecEnvs")

