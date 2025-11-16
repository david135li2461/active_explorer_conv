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


def load_policy_fallback(saved_path: str, env: ConvActiveExplorerEnv):
    if PPO is None:
        raise RuntimeError('stable-baselines3 not available in environment')
    if not zipfile.is_zipfile(saved_path):
        raise RuntimeError('Saved policy is not a zip archive')
    with zipfile.ZipFile(saved_path, 'r') as z:
        if 'policy.pth' not in z.namelist():
            raise RuntimeError('policy.pth not found in archive; cannot fallback')
        data = z.read('policy.pth')
    state = torch.load(io.BytesIO(data), map_location='cpu')
    model = PPO('CnnPolicy', env, verbose=0)
    try:
        model.policy.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'state_dict' in state:
            model.policy.load_state_dict(state['state_dict'])
        else:
            raise
    return model


def run_visualization(env: ConvActiveExplorerEnv, policy, episodes: int = 20, render_delay: float = 0.03):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError('cv2 not available; install opencv-python') from e

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=True)
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
    parser.add_argument('--threshold', type=float, default=0.5)
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
            policy = load_policy_fallback(args.saved_path, env)
    else:
        # fallback requires stable-baselines3; inform user
        raise RuntimeError('stable-baselines3 is not available in this environment')

    run_visualization(env, policy, episodes=args.episodes)


if __name__ == '__main__':
    main()
