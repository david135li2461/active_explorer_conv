"""Visualize a saved convolutional PPO policy for several episodes (cv2 render).
No CSV or logging produced — just render windows for human inspection.

Usage:
  conda run -n amnist-eval python3 active_explorer_conv/visualize_conv_policy.py \
    --saved-path active_explorer_conv/conv_ppo_200k/ppo_conv_explorer.zip \
    --classifier active_explorer_mnist/mnist_cnn.pth --episodes 20
"""
from __future__ import annotations

import argparse
import io
import zipfile
import time
import os
from typing import Optional

import torch
import numpy as np

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None

from conv_active_env import ConvActiveExplorerEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th
import gymnasium as gym


# SmallObsCNN mirrors the extractor used during training so fallback loading works
class SmallObsCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            cnn_out = self.cnn(sample)
            n_flatten = int(cnn_out.shape[1] * cnn_out.shape[2] * cnn_out.shape[3])
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.cnn(observations)
        return self.linear(x)

policy_kwargs = dict(features_extractor_class=SmallObsCNN, features_extractor_kwargs=dict(features_dim=256))


def load_policy_fallback(saved_path: str, env: ConvActiveExplorerEnv, policy_kwargs: dict | None = None):
    if PPO is None:
        raise RuntimeError('stable-baselines3 not available in environment')
    if not zipfile.is_zipfile(saved_path):
        raise RuntimeError('Saved policy is not a zip archive')
    with zipfile.ZipFile(saved_path, 'r') as z:
        if 'policy.pth' not in z.namelist():
            raise RuntimeError('policy.pth not found in archive; cannot fallback')
        data = z.read('policy.pth')
    state = torch.load(io.BytesIO(data), map_location='cpu')
    if policy_kwargs is None:
        model = PPO('CnnPolicy', env, verbose=0)
    else:
        model = PPO('CnnPolicy', env, verbose=0, policy_kwargs=policy_kwargs)
    try:
        model.policy.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'state_dict' in state:
            model.policy.load_state_dict(state['state_dict'])
        else:
            raise
    return model


def run_visualization(env: ConvActiveExplorerEnv, policy, episodes: int = 20, render_delay: float = 0.03, deterministic: bool = True):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError('cv2 not available; install opencv-python') from e

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, rew, terminated, truncated, info = env.step(int(action))
            env.render(mode='cv2')
            # small delay so humans can see frames
            time.sleep(render_delay)
        # short pause between episodes
        time.sleep(0.5)
    # cleanup windows
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved-path', type=str, required=True)
    parser.add_argument('--classifier', type=str, default='../active_explorer_mnist/mnist_cnn.pth')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--mode', type=str, choices=['deterministic', 'stochastic'], default='deterministic',
                        help='Run visualization deterministically or stochastically')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.saved_path):
        raise FileNotFoundError(args.saved_path)
    if not os.path.exists(args.classifier):
        raise FileNotFoundError(args.classifier)

    env = ConvActiveExplorerEnv(classifier_path=args.classifier, confidence_threshold=args.threshold, seed=args.seed)

    # try standard SB3 loader first
    policy = None
    if PPO is not None:
        try:
            policy = PPO.load(args.saved_path)
        except Exception as e:
            print('PPO.load failed, using fallback loader:', type(e).__name__, e)
            # ensure fallback instantiates matching architecture
            policy = load_policy_fallback(args.saved_path, env, policy_kwargs=policy_kwargs)
    else:
        # fallback requires stable-baselines3; inform user
        raise RuntimeError('stable-baselines3 is not available in this environment')

    deterministic_flag = True if args.mode == 'deterministic' else False
    run_visualization(env, policy, episodes=args.episodes, deterministic=deterministic_flag)


if __name__ == '__main__':
    main()
