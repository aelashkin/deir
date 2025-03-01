import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import FullyObsWrapper
from stable_baselines3.common.monitor import Monitor

# -------------------------
# 1. Define a simple embedding network.
# -------------------------
class SimpleEmbedding(nn.Module):
    def __init__(self, input_shape, embedding_dim=64):
        """
        A simple CNN that takes an image and outputs an embedding.
        Assumes input_shape is in (C, H, W) format.
        """
        super(SimpleEmbedding, self).__init__()
        c, h, w = input_shape
        self.embedding_dim = embedding_dim
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = 32 * h * w
        self.fc = nn.Linear(conv_out_size, embedding_dim)

    def forward(self, x):
        # Ensure x is batched: (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.conv(x)
        x = self.fc(x)
        return x

# -------------------------
# 2. Define a custom small CNN extractor for PPO.
# -------------------------
class SmallCnnExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        """
        Custom CNN extractor for small images.
        Expects observation_space of shape (C, H, W).
        """
        super(SmallCnnExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        # Two conv layers with kernel_size=3, stride=1 produce feature maps of size (H-2, W-2) then (H-4, W-4)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1),  # e.g., from 7x7 -> 5x5
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),                 # from 5x5 -> 3x3
            nn.ReLU()
        )
        n_flatten = 32 * 3 * 3
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        x = self.cnn(observations)
        x = th.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

# -------------------------
# 3. Create a custom DEIR wrapper.
# -------------------------
class DEIRWrapper(gym.Wrapper):
    def __init__(self, env, intrinsic_coef=0.1, embedding_dim=64):
        """
        This wrapper expects the underlying environment (from Farama MiniGrid)
        to have observations in a dict with an 'image' key in channel-last (H, W, C) format.
        It converts observations to channel-first format and computes an intrinsic reward
        using a fixed embedding network.
        """
        super(DEIRWrapper, self).__init__(env)
        self.intrinsic_coef = intrinsic_coef
        
        # If observation is a dict, extract the 'image' space.
        if isinstance(env.observation_space, gym.spaces.Dict):
            orig_space = env.observation_space.spaces['image']
            self.image_key = 'image'
        else:
            orig_space = env.observation_space
            self.image_key = None

        # Convert observation space from (H, W, C) to (C, H, W) if necessary.
        shape = orig_space.shape
        if len(shape) == 3 and shape[-1] in [1, 3]:
            new_shape = (shape[-1], shape[0], shape[1])
        else:
            new_shape = shape
        # Create a new Box observation space with channel-first shape.
        self.observation_space = gym.spaces.Box(
            low=orig_space.low.min(),
            high=orig_space.high.max(),
            shape=new_shape,
            dtype=orig_space.dtype,
        )

        # Create the fixed embedding network with the corrected input shape.
        self.embedding_net = SimpleEmbedding(self.observation_space.shape, embedding_dim)
        self.embedding_net.eval()
        for param in self.embedding_net.parameters():
            param.requires_grad = False

        # Episodic memory for storing embeddings.
        self.episodic_memory = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.image_key is not None:
            obs = obs[self.image_key]
        # Convert to float and normalize.
        obs = np.array(obs, dtype=np.float32) / 255.0
        # If observation is in channel-last, convert to channel-first.
        if obs.ndim == 3 and obs.shape[-1] in [1, 3]:
            obs = np.transpose(obs, (2, 0, 1))
        self.episodic_memory = []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.image_key is not None:
            obs = obs[self.image_key]
        obs = np.array(obs, dtype=np.float32) / 255.0
        if obs.ndim == 3 and obs.shape[-1] in [1, 3]:
            obs = np.transpose(obs, (2, 0, 1))
        image_tensor = th.tensor(obs, dtype=th.float32)
        with th.no_grad():
            embedding = self.embedding_net(image_tensor).squeeze(0)
        if len(self.episodic_memory) == 0:
            intrinsic_reward = 1.0
        else:
            distances = [th.norm(embedding - m).item() for m in self.episodic_memory]
            intrinsic_reward = min(distances)
        self.episodic_memory.append(embedding)
        total_reward = reward + self.intrinsic_coef * intrinsic_reward
        info['intrinsic_reward'] = intrinsic_reward
        return obs, total_reward, terminated, truncated, info

# -------------------------
# 4. Training and evaluation functions.
# -------------------------
def train_deir_agent(total_timesteps=10000, intrinsic_coef=0.1, embedding_dim=64):
    """
    Create the MiniGrid-Empty-Random-6x6-v0 env from Farama MiniGrid,
    wrap it with DEIRWrapper, and train a PPO agent.
    """
    # Create the env
    env = gym.make("MiniGrid-Empty-Random-6x6-v0")
    # Wrap with Monitor to record episode statistics
    env = Monitor(env)
    # Wrap with DEIRWrapper
    env = DEIRWrapper(env, intrinsic_coef=intrinsic_coef, embedding_dim=embedding_dim)
    # Vectorize the env
    env = DummyVecEnv([lambda: env])
    # Use our custom small CNN extractor to process the (3,7,7) observations.
    policy_kwargs = {
        "features_extractor_class": SmallCnnExtractor,
        "features_extractor_kwargs": {"features_dim": 64},
    }
    model = PPO("CnnPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model

def evaluate_deir_agent(model, num_episodes=100):
    """
    Evaluate the trained agent over a given number of episodes.
    """
    env = gym.make("MiniGrid-Empty-Random-6x6-v0")
    env = DEIRWrapper(env)
    rewards = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    avg_reward = np.mean(rewards)
    print("Average reward over {} episodes: {:.2f}".format(num_episodes, avg_reward))
    return rewards

# -------------------------
# 5. Main execution: train then evaluate.
# -------------------------
if __name__ == "__main__":
    model = train_deir_agent(total_timesteps=100000, intrinsic_coef=0.1, embedding_dim=64)
    evaluate_deir_agent(model, num_episodes=100)
