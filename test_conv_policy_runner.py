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
    # instantiate model with env to create the architecture
    model = PPO('CnnPolicy', env, verbose=0)
    try:
        model.policy.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'state_dict' in state:
            model.policy.load_state_dict(state['state_dict'])
        else:
            raise
    return model


def run_episode(env: ConvActiveExplorerEnv, policy, render: bool = False):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    moves = 0
    while not (terminated or truncated):
        if hasattr(policy, 'predict'):
            # policy is an SB3 model
            action, _ = policy.predict(obs, deterministic=True)
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
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num-episodes', type=int, default=100)
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
        for ep in range(args.num_episodes):
            res = run_episode(env, policy, render=args.render)
            row = {'episode': ep, **res}
            writer.writerow(row)
            print(f"Episode {ep}: true={res['true_label']} pred={res['predicted_label']} conf={res['confidence']:.3f} moves={res['moves']} pixels={res['pixels_seen']}")

    print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()
