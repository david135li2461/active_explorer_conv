"""Test a saved convolutional PPO policy on the ConvActiveExplorerEnv and log episodes to CSV.

CSV columns: episode, true_label, predicted_label, confidence, moves, pixels_seen

Usage example:
    conda run -n amnist-eval python3 active_explorer_conv/test_conv_policy_runner.py \
        --saved-path active_explorer_conv/conv_ppo/ppo_conv_explorer.zip \
        --num-episodes 100 --output active_explorer_conv/conv_policy_results.csv

By default rendering is off. Turn on with --render to see live cv2 windows (requires opencv-python).
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
import os
import time
from typing import Optional

import numpy as np
import torch

from conv_active_env import ConvActiveExplorerEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch as th
import gymnasium as gym

# SmallObsCNN matches the feature extractor used during training (3x28x28 -> features_dim)
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

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


def load_policy_fallback(saved_path: str, env: ConvActiveExplorerEnv):
    """Fallback loader: extract `policy.pth` from the saved SB3 zip and
    load it into a freshly-instantiated PPO('CnnPolicy', env).
    """
    if PPO is None:
        raise RuntimeError('stable-baselines3 not available in environment')
    if not zipfile.is_zipfile(saved_path):
        raise RuntimeError('Saved policy is not a zip archive')
    with zipfile.ZipFile(saved_path, 'r') as z:
        if 'policy.pth' not in z.namelist():
            raise RuntimeError('policy.pth not found in archive; cannot fallback')
        data = z.read('policy.pth')
    state = torch.load(io.BytesIO(data), map_location='cpu')
    # instantiate model with env to create the architecture (use same policy_kwargs as training)
    model = PPO('CnnPolicy', env, verbose=0, policy_kwargs=policy_kwargs)
    try:
        model.policy.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'state_dict' in state:
            model.policy.load_state_dict(state['state_dict'])
        else:
            raise
    return model


def _allowed_actions_from_conv_obs(env: ConvActiveExplorerEnv, obs: np.ndarray):
    """Return list of allowed actions (0=up,1=down,2=left,3=right) according to FloodPolicy rules for conv obs.

    - prefer unexplored neighbors if any, otherwise allow any valid neighbor
    """
    # obs channels: 0=mask,1=partial,2=loc
    mask = obs[0]
    loc = obs[2]
    # find position
    ys, xs = np.where(loc >= 0.5)
    if len(ys) == 0:
        # fallback: choose center
        r, c = env.img_h // 2, env.img_w // 2
    else:
        r, c = int(ys[0]), int(xs[0])

    neighbors = []  # (action, nr, nc)
    if r - 1 >= 0:
        neighbors.append((0, r - 1, c))
    if r + 1 < env.img_h:
        neighbors.append((1, r + 1, c))
    if c - 1 >= 0:
        neighbors.append((2, r, c - 1))
    if c + 1 < env.img_w:
        neighbors.append((3, r, c + 1))

    unexplored = [a for (a, nr, nc) in neighbors if mask[nr, nc] < 0.5]
    if len(unexplored) > 0:
        return [int(a) for a in unexplored]
    return [int(a) for (a, _, _) in neighbors]


def _get_action_probs_from_sb3(policy, obs: np.ndarray):
    """Try to extract the action probability vector from an SB3 policy for a single observation.

    Returns a 1D numpy array of length `action_space.n` or None if not possible.
    """
    try:
        obs_t = th.as_tensor(obs[None]).float()
        # try common SB3 policy API
        if hasattr(policy.policy, 'get_distribution'):
            dist = policy.policy.get_distribution(obs_t)
            # distribution wrappers vary; try several attributes
            if hasattr(dist, 'distribution') and hasattr(dist.distribution, 'probs'):
                probs = dist.distribution.probs.detach().cpu().numpy()[0]
                return probs
            if hasattr(dist, 'probs'):
                probs = dist.probs.detach().cpu().numpy()[0]
                return probs
            if hasattr(dist, 'logits'):
                logits = dist.logits.detach().cpu().numpy()[0]
                exp = np.exp(logits - np.max(logits))
                return exp / exp.sum()
        # fallback: try to call forward-like API
        if hasattr(policy.policy, 'forward'):
            out = policy.policy.forward(obs_t)
            # unknown shape/contents; give up
    except Exception:
        pass
    return None


def run_episode(env: ConvActiveExplorerEnv, policy, render: bool = False, deterministic: bool = True, mask_with_flood: bool = False):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    moves = 0
    while not (terminated or truncated):
        if hasattr(policy, 'predict'):
            # policy is an SB3 model
            if (not deterministic) and mask_with_flood:
                # compute allowed actions under FloodPolicy-like local rules
                allowed = _allowed_actions_from_conv_obs(env, obs)
                probs = _get_action_probs_from_sb3(policy, obs)
                if probs is not None and len(probs) >= 1:
                    probs_masked = probs.copy()
                    for i in range(len(probs_masked)):
                        if i not in allowed:
                            probs_masked[i] = 0.0
                    s = float(probs_masked.sum())
                    if s <= 0.0:
                        # no probability mass left; fall back to uniform over allowed
                        action = int(np.random.choice(allowed))
                    else:
                        probs_masked = probs_masked / s
                        action = int(np.random.choice(len(probs_masked), p=probs_masked))
                else:
                    # fallback: repeatedly sample until we hit an allowed action (limited tries)
                    action = None
                    for _ in range(200):
                        cand, _ = policy.predict(obs, deterministic=False)
                        if int(cand) in allowed:
                            action = int(cand)
                            break
                    if action is None:
                        action = int(np.random.choice(allowed))
            else:
                action, _ = policy.predict(obs, deterministic=deterministic)
        else:
            raise RuntimeError('Unsupported policy type for conv runner')
        obs, rew, terminated, truncated, info = env.step(int(action))
        moves += 1
        if render:
            env.render(mode='cv2')
            time.sleep(0.02)
        if moves > env.max_steps + 10:
            break

    pixels_seen = int(info.get('pixels_seen', int(env._mask.sum()) if getattr(env, '_mask', None) is not None else 0))
    moves_taken = int(info.get('moves', moves))
    return {
        'true_label': int(info.get('true_label', -1)),
        'predicted_label': int(info.get('classifier_pred', -1)),
        'confidence': float(info.get('classifier_max_confidence', 0.0)),
        'moves': moves_taken,
        'pixels_seen': pixels_seen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved-path', type=str, required=True)
    parser.add_argument('--classifier', type=str, default='../active_explorer_mnist/mnist_cnn.pth')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--mode', type=str, choices=['deterministic', 'stochastic'], default='deterministic',
                        help='Whether to run the policy deterministically or stochastically')
    parser.add_argument('--mask-with-flood', action='store_true',
                        help='When sampling stochastically, restrict actions to those allowed by FloodPolicy and renormalize')
    parser.add_argument('--output', type=str, default='conv_policy_results.csv')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.saved_path is None or not os.path.exists(args.saved_path):
        raise FileNotFoundError('--saved-path must point to an existing SB3 .zip file')

    env = ConvActiveExplorerEnv(classifier_path=args.classifier, confidence_threshold=args.threshold, seed=args.seed)

    if PPO is None:
        raise RuntimeError('stable-baselines3 not available; install it to use a saved policy')

    try:
        policy = PPO.load(args.saved_path)
    except Exception as e:
        print('PPO.load failed, falling back to PyTorch state dict loader:', type(e).__name__, e)
        policy = load_policy_fallback(args.saved_path, env)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['episode', 'true_label', 'predicted_label', 'confidence', 'moves', 'pixels_seen'])
        writer.writeheader()
        deterministic_flag = True if args.mode == 'deterministic' else False
        for ep in range(args.num_episodes):
            res = run_episode(env, policy, render=args.render, deterministic=deterministic_flag,
                              mask_with_flood=args.mask_with_flood)
            row = {'episode': ep, **res}
            writer.writerow(row)
            print(f"Episode {ep}: true={res['true_label']} pred={res['predicted_label']} conf={res['confidence']:.3f} moves={res['moves']} pixels={res['pixels_seen']}")

    print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()
