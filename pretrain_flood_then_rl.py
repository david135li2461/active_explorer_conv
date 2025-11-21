"""Behaviorally clone FloodPolicyConv to initialize PPO CnnPolicy, then continue RL training.

This mirrors `active_explorer_mnist/pretrain_flood_then_rl.py` but for conv observations.

Usage (dry-run):
    python pretrain_flood_then_rl.py --classifier ../active_explorer_mnist/mnist_cnn.pth --bc-samples 2000 --bc-epochs 3 --rl-timesteps 0

To run full training from Flood init for 300k timesteps:
    python pretrain_flood_then_rl.py --classifier ../active_explorer_mnist/mnist_cnn.pth --bc-samples 20000 --bc-epochs 5 --rl-timesteps 300000 --save-dir ./conv_ppo_bc
"""
from __future__ import annotations

import argparse
import io
import zipfile
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import gymnasium as gym

from conv_active_env import ConvActiveExplorerEnv
# reuse FloodPolicyConv from test_conv_policy_runner
from test_conv_policy_runner import FloodPolicyConv


class Uint8ImageWrapper(gym.ObservationWrapper):
    """Convert float [0,1] channel-first images to uint8 [0,255] as expected by SB3's default CNN feature extractor.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        low = np.zeros(obs_space.shape, dtype=np.uint8)
        high = np.ones(obs_space.shape, dtype=np.uint8) * 255
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)

    def observation(self, obs):
        img = (obs * 255.0).clip(0, 255).astype(np.uint8)
        return img


# Feature extractor used during training; matches train_conv_explorer.SmallObsCNN
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


def make_env(classifier_path: str, seed: int = 0, threshold: float = 0.9):
    def _thunk():
        env = ConvActiveExplorerEnv(classifier_path=classifier_path, confidence_threshold=threshold, seed=seed)
        env = Uint8ImageWrapper(env)
        env = Monitor(env)
        return env
    return _thunk


def collect_bc_data(env: ConvActiveExplorerEnv, n_samples: int) -> (np.ndarray, np.ndarray):
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    policy = FloodPolicyConv(env)
    collected = 0
    while collected < n_samples:
        o, _ = env.reset()
        done = False
        while not done and collected < n_samples:
            a, _ = policy.predict(o)
            obs_list.append(o.copy())
            act_list.append(int(a))
            o, rew, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            collected += 1
    return np.stack(obs_list, axis=0), np.array(act_list, dtype=np.int64)


def behavioral_clone(model: PPO, obs: np.ndarray, acts: np.ndarray, epochs: int = 5, batch_size: int = 64, lr: float = 1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.policy.to(device)
    dataset = TensorDataset(torch.from_numpy(obs).float(), torch.from_numpy(acts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)

    for ep in range(epochs):
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            # The policy's feature extractor expects inputs like the wrapped env provides.
            # Here we directly call extract_features + mlp_extractor as in the flat script.
            with torch.no_grad():
                features = model.policy.extract_features(xb)
            latent_pi, _ = model.policy.mlp_extractor(features)
            dist = model.policy._get_action_dist_from_latent(latent_pi)
            logp = dist.log_prob(yb)
            if logp.dim() > 1:
                logp = logp.sum(dim=1)
            loss = -logp.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        print(f"BC epoch {ep+1}/{epochs}: loss={total_loss/n:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', type=str, default='../active_explorer_mnist/mnist_cnn.pth')
    parser.add_argument('--bc-samples', type=int, default=20000)
    parser.add_argument('--bc-epochs', type=int, default=5)
    parser.add_argument('--bc-batch', type=int, default=64)
    parser.add_argument('--rl-timesteps', type=int, default=300000)
    parser.add_argument('--save-dir', type=str, default='./conv_ppo_bc')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # data collector env (no uint8 wrapper; FloodPolicyConv expects raw float obs)
    env0 = ConvActiveExplorerEnv(classifier_path=args.classifier, confidence_threshold=args.threshold, seed=args.seed)
    print(f"Collecting {args.bc_samples} BC samples using FloodPolicyConv...")
    obs, acts = collect_bc_data(env0, args.bc_samples)
    print("Collected", obs.shape[0], "samples")

    # training env (wrapped like during original training)
    vec = DummyVecEnv([make_env(args.classifier, seed=args.seed, threshold=args.threshold)])
    model = PPO('CnnPolicy', vec, verbose=1, policy_kwargs=policy_kwargs, seed=args.seed)

    print("Running behavioral cloning to initialize policy...")
    behavioral_clone(model, obs, acts, epochs=args.bc_epochs, batch_size=args.bc_batch, lr=args.lr)

    # save intermediate model
    pre_path = os.path.join(args.save_dir, 'ppo_conv_pretrained')
    model.save(pre_path)
    print("Saved pretrained model to", pre_path + '.zip')

    if args.rl_timesteps > 0:
        # Use a chunked learn loop so we can present a live tqdm bar similar
        # to `train_conv_explorer.py` without adding a callback dependency.
        print(f"Continuing RL for {args.rl_timesteps} timesteps")
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        total = int(args.rl_timesteps)
        chunk = max(2000, total // 50)
        learned = 0
        if tqdm is None:
            # fallback: single call
            model.learn(total_timesteps=total)
            learned = total
        else:
            with tqdm(total=total, desc='RL training', unit='steps') as pbar:
                while learned < total:
                    this_chunk = min(chunk, total - learned)
                    model.learn(total_timesteps=this_chunk, reset_num_timesteps=False)
                    learned += this_chunk
                    pbar.update(this_chunk)

        final_path = os.path.join(args.save_dir, 'ppo_conv_final')
        model.save(final_path)
        print("Saved final model to", final_path + '.zip')
    else:
        print("Skipping RL (rl-timesteps=0). Done.")


if __name__ == '__main__':
    main()
