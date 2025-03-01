# File: train_deir_minigrid.py

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch as th
import numpy as np

import os
import sys

# Add project root directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Then keep your existing PYTHONPATH code as a fallback
pythonpath = os.getenv("PYTHONPATH")
if (pythonpath and pythonpath not in sys.path):
    sys.path.append(pythonpath)

# Import our custom PPOTrainer and PPOModel from the repository
from src.algo.ppo_trainer import PPOTrainer
from src.algo.ppo_model import PPOModel
from src.utils.loggers import StatisticsLogger, LocalLogger
from src.algo.ppo_rollout import PPORollout
# (Assume other necessary imports from the repo are available.)

# Replace gym_minigrid imports with minigrid
from minigrid.envs import DoorKeyEnv  # Instead of from gym_minigrid.envs
from minigrid.core.grid import Grid  # Instead of from gym_minigrid.minigrid
from minigrid.core.world_object import Door, Key, Wall, Goal  # Instead of from gym_minigrid.minigrid
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC  # Instead of from gym_minigrid.minigrid

# Instead of gym_minigrid.register, use gymnasium registry
from gymnasium.envs.registration import register

# Ensure SB3 knows we're using Gymnasium
import stable_baselines3.common.base_class
stable_baselines3.common.base_class.is_gymnasium_available = lambda: True

import stable_baselines3
print(f"Using Stable Baselines3 version: {stable_baselines3.__version__}")

# Add this with your other imports
from src.utils.enum_types import ModelType

# --- HELPER FUNCTIONS ---


# Add this function to the existing file

def obs_as_tensor(obs, device):
    """
    Convert an observation from numpy arrays to PyTorch tensors.
    
    Args:
        obs: Observation (numpy array, dict of numpy arrays, or other formats)
        device: PyTorch device to put the tensors on
        
    Returns:
        Tensor or dict of tensors
    """
    if isinstance(obs, dict):
        return {k: obs_as_tensor(v, device) for k, v in obs.items()}
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs).to(device)
    return th.as_tensor(obs).to(device)


# --- MODIFIED FUNCTIONS ---

class GymnasiumPPOTrainer(PPOTrainer):
    """
    A subclass of our custom PPOTrainer that adapts collect_rollouts (and initial reset)
    to the gymnasium v1.0.0 API.
    """
    def collect_rollouts(self, env, callback, ppo_rollout_buffer, n_rollout_steps):
        """
        Overridden to handle gymnasium's (obs, reward, terminated, truncated, info)
        step API.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        ppo_rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()
        self.init_on_rollout_start()
        while n_steps < n_rollout_steps:
            print(f"[DEBUG] Starting rollout step {n_steps}")
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)
            with th.no_grad():
                # Convert last observation to tensor (using the repo’s obs_as_tensor utility)
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs, policy_mems = self.policy.forward(obs_tensor, self._last_policy_mems)
                actions = actions.cpu().numpy()
                print(f"[DEBUG] Actions computed: {actions}")
            clipped_actions = actions
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            self.log_before_transition(values)
            # ---- Adapted step call for gymnasium ----
            new_obs, rewards, terminated, truncated, infos = env.step(clipped_actions)
            print(f"[DEBUG] Received new_obs: {new_obs}, rewards: {rewards}")
            dones = np.logical_or(terminated, truncated)
            # (Optionally, if observations are dictionaries, extract the "rgb" key)
            if isinstance(new_obs, dict) and "rgb" in new_obs:
                new_obs = new_obs["rgb"]
            if self.env_render:
                env.render()
            with th.no_grad():
                new_obs_tensor = obs_as_tensor(new_obs, self.device)
                _, new_values, _, _ = self.policy.forward(new_obs_tensor, policy_mems)
            intrinsic_rewards, model_mems = self.create_intrinsic_rewards(new_obs, actions, dones)
            self.log_after_transition(rewards, intrinsic_rewards)
            self.clear_on_episode_end(dones, policy_mems, model_mems)
            self.num_timesteps += env.num_envs
            self._update_info_buffer(infos)
            n_steps += 1
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape(-1, 1)
            ppo_rollout_buffer.add(
                self._last_obs,
                new_obs,
                self._last_policy_mems,
                self._last_model_mems,
                actions,
                rewards,
                intrinsic_rewards,
                self._last_episode_starts,
                dones,
                values,
                log_probs,
                self.curr_key_status,
                self.curr_door_status,
                self.curr_target_dists,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            if policy_mems is not None:
                self._last_policy_mems = policy_mems.detach().clone()
            if model_mems is not None:
                self._last_model_mems = model_mems.detach().clone()
        ppo_rollout_buffer.compute_intrinsic_rewards()
        ppo_rollout_buffer.compute_returns_and_advantage(new_values, dones)
        callback.on_rollout_end()
        return True

    def _setup_initial_obs(self, env):
        """
        Adapts the initial reset call to the gymnasium API.
        """
        self._last_obs, info = env.reset()
        print(f"[DEBUG] Initial observation set up: {self._last_obs}")
        self._last_episode_starts = np.zeros((env.num_envs,), dtype=bool)

    def __init__(
        self,
        policy,
        env,
        run_id,  # Add this
        learning_rate=3e-4,
        model_learning_rate=3e-4,  # Add this
        n_steps=2048,  # Add this
        batch_size=64,  # Add this
        n_epochs=4,  # Add this
        model_n_epochs=4,  # Add this
        gamma=0.99,  # Add this
        gae_lambda=0.95,  # Add this
        clip_range=0.2,  # Add this
        clip_range_vf=None,  # Add this
        ent_coef=0.0,  # Add this
        pg_coef=1.0,  # Add this
        vf_coef=0.5,  # Add this
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,  # Add this
        int_rew_source=ModelType.DEIR,  # Add this
        int_rew_norm=0,  # Add this
        int_rew_coef=1e-3,  # Add this
        int_rew_momentum=1.0,  # Add this
        int_rew_eps=0.0,  # Add this
        int_rew_clip=0.0,  # Add this
        adv_momentum=0.0,  # Add this
        image_noise_scale=0.0,  # Add this
        enable_plotting=0,  # Add this
        can_see_walls=1,  # Add this
        ext_rew_coef=1.0,  # Add this
        adv_norm=1,  # Add this
        adv_eps=1e-8,  # Add this
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        env_source=None,  # Add this
        env_render=None,  # Add this
        fixed_seed=None,  # Add this
        plot_interval=10,  # Add this
        plot_colormap='Blues',  # Add this
        log_explored_states=None,  # Add this
        local_logger=None,  # Add this
        use_wandb=False,  # Add this
    ):
        super(GymnasiumPPOTrainer, self).__init__(
            policy=policy,
            env=env,
            run_id=run_id,  # Add this
            learning_rate=learning_rate,
            model_learning_rate=model_learning_rate,  # Add this
            n_steps=n_steps,  # Add this
            batch_size=batch_size,  # Add this
            n_epochs=n_epochs,  # Add this
            model_n_epochs=model_n_epochs,  # Add this
            gamma=gamma,  # Add this
            gae_lambda=gae_lambda,  # Add this
            clip_range=clip_range,  # Add this
            clip_range_vf=clip_range_vf,  # Add this
            ent_coef=ent_coef,  # Add this
            pg_coef=pg_coef,  # Add this
            vf_coef=vf_coef,  # Add this
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,  # Add this
            int_rew_source=int_rew_source,  # Add this
            int_rew_norm=int_rew_norm,  # Add this
            int_rew_coef=int_rew_coef,  # Add this
            int_rew_momentum=int_rew_momentum,  # Add this
            int_rew_eps=int_rew_eps,  # Add this
            int_rew_clip=int_rew_clip,  # Add this
            adv_momentum=adv_momentum,  # Add this
            image_noise_scale=image_noise_scale,  # Add this
            enable_plotting=enable_plotting,  # Add this
            can_see_walls=can_see_walls,  # Add this
            ext_rew_coef=ext_rew_coef,  # Add this
            adv_norm=adv_norm,  # Add this
            adv_eps=adv_eps,  # Add this
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            env_source=env_source,  # Add this
            env_render=env_render,  # Add this
            fixed_seed=fixed_seed,  # Add this
            plot_interval=plot_interval,  # Add this
            plot_colormap=plot_colormap,  # Add this
            log_explored_states=log_explored_states,  # Add this
            local_logger=local_logger,  # Add this
            use_wandb=use_wandb,  # Add this
            # env_type="Gymnasium"  # This was already there
        )

# --- MAIN TRAINING/EVALUATION CODE ---

def main():
    total_timesteps = 100000
    eval_episodes = 100
    env_id = "MiniGrid-Empty-Random-6x6-v0"

    # Create a vectorized environment using SB3's DummyVecEnv
    def make_env():
        def _init():
            env = gym.make(env_id)
            return env
        return _init
    num_envs = 1  # single env for training
    
    # Replace SyncVectorEnv with DummyVecEnv
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
    
    # Rest of your code remains the same...
    model = GymnasiumPPOTrainer(
        policy=PPOModel,
        env=vec_env,
        run_id=0,
        learning_rate=3e-4,
        model_learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        model_n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        pg_coef=1.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        int_rew_source=ModelType.DEIR,  # Using the enum directly
        int_rew_coef=1e-2,
        int_rew_norm=1,
        int_rew_momentum=0.9,
        int_rew_eps=1e-5,
        int_rew_clip=0,
        adv_momentum=0.9,
        image_noise_scale=0.0,
        enable_plotting=0,
        can_see_walls=1,
        ext_rew_coef=1.0,
        adv_norm=2,
        adv_eps=1e-8,
        device="auto",
        env_source=0,  # EnvSrc.MiniGrid is 0
        env_render=0,
        fixed_seed=-1,
        plot_interval=10,
        plot_colormap="Blues",
        log_explored_states=0,
        use_wandb=False,
    )
    # Set up the initial observation using the adapted reset API.
    model._setup_initial_obs(vec_env)

    # Train the agent.
    model.learn(total_timesteps=total_timesteps)

    # --- Evaluation loop over 100 episodes using gymnasium API ---
    eval_env = gym.make(env_id)
    episode_rewards = []
    for episode in range(eval_episodes):
        print(f"[DEBUG] Starting evaluation episode {episode}")
        obs, info = eval_env.reset()
        done = False
        total_reward = 0.0
        # For vectorized envs you might need to adjust, but here we assume a single env.
        while not done:
            # Get action from the trained policy (convert observation to tensor)
            obs_tensor = obs_as_tensor(obs, model.device)
            with th.no_grad():
                action, _, _, _ = model.policy.forward(obs_tensor, model._last_policy_mems)
            print(f"[DEBUG] Evaluation action: {action.cpu().numpy()}")
            action = action.cpu().numpy()
            # Gymnasium step returns (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            print(f"[DEBUG] Evaluation step reward: {reward}")
            done = terminated or truncated
            total_reward += reward
        episode_rewards.append(total_reward)
    avg_reward = np.mean(episode_rewards)
    print(f"Average reward over {eval_episodes} evaluation episodes: {avg_reward}")

if __name__ == "__main__":
    main()
