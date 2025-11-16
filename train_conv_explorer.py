"""Train a PPO agent on the convolutional-observation Active MNIST env.

This script creates a DummyVecEnv wrapper, trains for a configurable number
of timesteps using `CnnPolicy`, and saves the model.
"""
from __future__ import annotations

import argparse
import os
from functools import partial

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import io
import zipfile
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from conv_active_env import ConvActiveExplorerEnv
import sys


class Uint8ImageWrapper(gym.ObservationWrapper):
    """Convert float [0,1] channel-first images to uint8 [0,255] as expected
    by SB3's default CNN feature extractor (NatureCNN).
    """
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box)
        low = np.zeros(obs_space.shape, dtype=np.uint8)
        high = np.ones(obs_space.shape, dtype=np.uint8) * 255
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.uint8)

    def observation(self, obs):
        # obs assumed float32 in [0,1]
        img = (obs * 255.0).clip(0, 255).astype(np.uint8)
        return img


def make_env(classifier_path: str, seed: int = 0):
    def _thunk():
        env = ConvActiveExplorerEnv(classifier_path=classifier_path, confidence_threshold=0.5, seed=seed)
        # convert observation to uint8 0-255 channels-first image expected by SB3's CNN
        env = Uint8ImageWrapper(env)
        env = Monitor(env)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, default='../active_explorer_mnist/mnist_cnn.pth')
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--save-dir', type=str, default='./conv_ppo')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Path to an SB3 .zip checkpoint to resume from (supports fallback)')
    parser.add_argument('--lr', type=float, default=None, help='Optional new learning rate when resuming')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    env = DummyVecEnv([make_env(args.classifier, seed=args.seed)])

    # Define a small custom CNN feature extractor compatible with 3x28x28 inputs
    class SmallObsCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
            # observation_space.shape == (3, 28, 28)
            super().__init__(observation_space, features_dim)
            n_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            # compute shape after cnn
            with th.no_grad():
                sample = th.as_tensor(observation_space.sample()[None]).float()
                cnn_out = self.cnn(sample)
                n_flatten = int(cnn_out.shape[1] * cnn_out.shape[2] * cnn_out.shape[3])
            self.linear = nn.Sequential(nn.Flatten(), nn.Linear(n_flatten, features_dim), nn.ReLU())

        def forward(self, observations: th.Tensor) -> th.Tensor:
            x = self.cnn(observations)
            return self.linear(x)

    policy_kwargs = dict(features_extractor_class=SmallObsCNN, features_extractor_kwargs=dict(features_dim=256))

    # Helper: robust load with fallback to policy.pth state dict extraction
    def load_with_fallback(saved_path: str):
        try:
            print('Trying stable-baselines3 PPO.load()')
            m = PPO.load(saved_path, env=env, policy_kwargs=policy_kwargs, verbose=1)
            print('Loaded with PPO.load()')
            return m
        except Exception as e:
            print('PPO.load failed:', type(e).__name__, e)
            print('Falling back to state-dict extraction from zip...')
        if not zipfile.is_zipfile(saved_path):
            raise RuntimeError('Saved model is not a zip archive')
        with zipfile.ZipFile(saved_path, 'r') as z:
            if 'policy.pth' not in z.namelist():
                raise RuntimeError('policy.pth not found in archive; cannot fallback')
            data = z.read('policy.pth')
        state = th.load(io.BytesIO(data), map_location='cpu')
        # instantiate fresh model
        m = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
        try:
            m.policy.load_state_dict(state)
        except Exception:
            m.policy.load_state_dict(state.get('state_dict', state))
        return m

    model = None
    if args.resume_from:
        if not os.path.exists(args.resume_from):
            raise FileNotFoundError(f'--resume-from path not found: {args.resume_from}')
        model = load_with_fallback(args.resume_from)
        # optionally set a smaller LR for fine-tuning
        if args.lr is not None:
            print('Setting resumed model learning_rate to', args.lr)
            try:
                model.learning_rate = args.lr
            except Exception:
                pass

    if model is None:
        model = PPO('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)


    # Progress bar callback using tqdm
    class TqdmCallback(BaseCallback):
        """Simple tqdm progress bar callback that tracks `num_timesteps`."""
        def __init__(self, total_timesteps: int):
            super().__init__(verbose=0)
            self.total = int(total_timesteps)
            self.pbar = None
            self._last = 0

        def _on_training_start(self) -> None:
            try:
                # create lazily to avoid hard dependency at import time
                from tqdm import tqdm
            except Exception:
                self.pbar = None
                return
            # use stdout and dynamic width so the bar updates correctly in different terminals
            self.pbar = tqdm(total=self.total, desc='training', file=sys.stdout, dynamic_ncols=True, mininterval=0.1)
            self._last = 0

        def _on_step(self) -> bool:
            if self.pbar is None:
                return True
            n = int(self.num_timesteps)
            delta = n - self._last
            if delta > 0:
                self.pbar.update(delta)
                self._last = n
                try:
                    # ensure immediate refresh
                    self.pbar.refresh()
                except Exception:
                    pass
            return True

        def _on_training_end(self) -> None:
            if self.pbar is not None:
                self.pbar.close()

    # Use only tqdm progress bar (no intermediate checkpointing)
    tqdm_cb = TqdmCallback(total_timesteps=args.timesteps)
    # If resuming from checkpoint we should not reset the internal timestep counter
    reset_num = True if args.resume_from is None else False
    model.learn(total_timesteps=args.timesteps, callback=tqdm_cb, reset_num_timesteps=reset_num)

    # final save
    save_path = os.path.join(args.save_dir, 'ppo_conv_explorer.zip')
    model.save(save_path)
    print('Saved model to', save_path)


if __name__ == '__main__':
    main()
