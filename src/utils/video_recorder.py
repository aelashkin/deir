import os
from typing import Callable

from gymnasium.wrappers.rendering import RecordVideo
import gymnasium as gym  # For gym.Env

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


# Helper class for RecordVideo to wrap the VecEnv's rendering capability
class _RenderProxyEnv(gym.Env):
    def __init__(self, vec_env: VecEnv, render_fps: int):
        super().__init__()
        self.vec_env = vec_env
        self.render_mode = "rgb_array"  # RecordVideo expects a render_mode that produces frames
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": render_fps}

        # gymnasium.Env requires action_space and observation_space to be set.
        # Use the spaces from the wrapped VecEnv.
        self.action_space = self.vec_env.action_space
        self.observation_space = self.vec_env.observation_space

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = self.observation_space.sample()
        info = {}
        return obs, info

    def render(self):
        return self.vec_env.render()

    def close(self):
        pass


class VecVideoRecorder(VecEnvWrapper):
    """
    Wraps a VecEnv or VecEnvWrapper object to record rendered images as mp4 video.
    It uses gymnasium.wrappers.rendering.RecordVideo for the recording process.

    :param venv: The vectorized environment to wrap.
    :param video_folder: Where to save videos.
    :param record_video_trigger: Function that defines when to start recording.
                                 The function takes the current number of steps
                                 and returns whether we should start recording or not.
    :param video_length: Length of recorded videos in frames.
    :param name_prefix: Prefix for the video file name.
    """

    def __init__(
        self,
        venv: VecEnv,
        video_folder: str,
        record_video_trigger: Callable[[int], bool],
        video_length: int = 200,
        name_prefix: str = "rl-video",
    ):
        VecEnvWrapper.__init__(self, venv)

        self.fps = 30  # Default FPS

        _current_env_for_metadata = self.venv
        while isinstance(_current_env_for_metadata, VecEnvWrapper):
            _current_env_for_metadata = _current_env_for_metadata.venv

        env_metadata = None
        if hasattr(self.venv, 'metadata') and self.venv.metadata is not None:
            env_metadata = self.venv.metadata

        if not env_metadata and hasattr(self.venv, 'get_attr'):
            try:
                all_metadata = self.venv.get_attr("metadata")
                if all_metadata and len(all_metadata) > 0 and all_metadata[0] is not None:
                    env_metadata = all_metadata[0]
            except Exception:
                pass

        if not env_metadata and hasattr(_current_env_for_metadata, 'metadata') and _current_env_for_metadata.metadata is not None:
            env_metadata = _current_env_for_metadata.metadata

        if env_metadata and 'render_fps' in env_metadata:
            self.fps = env_metadata['render_fps']

        self.record_video_trigger = record_video_trigger
        self.video_folder = os.path.abspath(video_folder)
        os.makedirs(self.video_folder, exist_ok=True)

        self.name_prefix = name_prefix
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.current_video_segment_name: str | None = None

        self.gymnasium_recorder: RecordVideo | None = None
        self.proxy_render_env = _RenderProxyEnv(self.venv, render_fps=self.fps)

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset()
        if self._video_enabled() and not self.recording:
            self.start_video_recorder()

        if self.gymnasium_recorder and self.gymnasium_recorder.recording:
            self.gymnasium_recorder.reset(seed=self.step_id)
        return obs

    def start_video_recorder(self) -> None:
        self.close_video_recorder()

        self.current_video_segment_name = (
            f"{self.name_prefix}-step-{self.step_id}-to-step-{self.step_id + self.video_length}"
        )

        self.gymnasium_recorder = RecordVideo(
            env=self.proxy_render_env,
            video_folder=self.video_folder,
            name_prefix=self.name_prefix,
            video_length=self.video_length,
            fps=self.fps,
            episode_trigger=lambda _: False,
            step_trigger=lambda _: False,
            disable_logger=True,
        )

        self.gymnasium_recorder.start_recording(self.current_video_segment_name)
        self.recording = True

    def _video_enabled(self) -> bool:
        return self.record_video_trigger(self.step_id)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.venv.step_wait()
        self.step_id += 1

        if self.recording:
            if self.gymnasium_recorder:
                dummy_action = self.proxy_render_env.action_space.sample()
                self.gymnasium_recorder.step(dummy_action)

                if not self.gymnasium_recorder.recording:
                    self.close_video_recorder()
            else:
                self.recording = False
        elif self._video_enabled():
            self.start_video_recorder()
            if self.gymnasium_recorder and self.gymnasium_recorder.recording:
                dummy_action = self.proxy_render_env.action_space.sample()
                self.gymnasium_recorder.step(dummy_action)

        return obs, rews, dones, infos

    def close_video_recorder(self) -> None:
        if self.gymnasium_recorder is not None:
            if self.gymnasium_recorder.recording:
                if self.current_video_segment_name:
                    expected_path = os.path.join(self.video_folder, f"{self.current_video_segment_name}.mp4")
                    print(f"Saving video to {expected_path}")

                self.gymnasium_recorder.stop_recording()

            self.gymnasium_recorder.close()
            self.gymnasium_recorder = None

        self.recording = False
        self.current_video_segment_name = None

    def close(self) -> None:
        super().close()
        self.close_video_recorder()

    def __del__(self):
        self.close()
