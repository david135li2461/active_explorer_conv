"""Convolutional-observation Active MNIST environment.

Observation: 3 x 28 x 28 (channels-first)
 - channel 0: mask (1 where revealed, 0 otherwise)
 - channel 1: partial image values in [0,1] (0 for unrevealed)
 - channel 2: agent location indicator (1 at current pixel, 0 elsewhere)

Action space: Discrete(4) (up, down, left, right)
Termination: when classifier's max softmax >= confidence_threshold (reward=1), or after max_steps (truncated).

This env re-uses the `SmallCNN` classifier architecture defined in
`active_explorer_mnist/active_mnist_env.py` and expects a compatible
state dict (e.g. `active_explorer_mnist/mnist_cnn.pth`).
"""
from __future__ import annotations

import os
import typing as t
import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

"""
Include SmallCNN definition here (copied from the existing active_mnist_env
so this module is self-contained and doesn't require importing the other
package path). This ensures the classifier architecture is identical to the
one used elsewhere in the project.
"""
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, padding: int = 0):
        super().__init__()
        self.padding = int(padding)
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=self.padding)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=self.padding)
        self.pool = nn.MaxPool2d(2, 2)
        s = 28
        s = s + 2 * self.padding - 2
        s = s + 2 * self.padding - 2
        s = s // 2
        fc_in = 64 * s * s
        self.fc1 = nn.Linear(fc_in, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvActiveExplorerEnv(gym.Env):
    metadata = {"render_modes": ["human", "cv2"]}

    def __init__(
        self,
        classifier_path: t.Optional[str] = None,
        confidence_threshold: float = 0.5,
        max_steps: int = 500,
        seed: t.Optional[int] = None,
        mnist_root: str = "./data",
    ) -> None:
        super().__init__()

        assert 0.0 < confidence_threshold < 1.0, "confidence_threshold must be in (0,1)"
        self.confidence_threshold = float(confidence_threshold)
        self.max_steps = int(max_steps)
        self.seed(seed)

        # action space
        self.action_space = spaces.Discrete(4)

        # observation: channels-first (3, 28, 28)
        self.img_h = 28
        self.img_w = 28
        self.num_channels = 3
        low = np.zeros((self.num_channels, self.img_h, self.img_w), dtype=np.float32)
        high = np.ones((self.num_channels, self.img_h, self.img_w), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # MNIST dataset
        self.mnist_transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_root = mnist_root
        try:
            self._mnist = datasets.MNIST(self.mnist_root, train=True, download=True, transform=self.mnist_transform)
        except Exception as e:
            warnings.warn(f"Failed to download/load MNIST dataset at {self.mnist_root}: {e}")
            self._mnist = None

        # classifier model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = SmallCNN(num_classes=10).to(self.device)
        loaded = False
        if classifier_path is not None:
            if os.path.exists(classifier_path):
                self._load_classifier_state(classifier_path)
                loaded = True
            else:
                raise FileNotFoundError(f"classifier_path provided but file not found: {classifier_path}")
        else:
            # try to find existing mnist weights in the sibling folder
            candidate = os.path.join(os.path.dirname(__file__), '..', 'active_explorer_mnist', 'mnist_cnn.pth')
            candidate = os.path.normpath(candidate)
            if os.path.exists(candidate):
                self._load_classifier_state(candidate)
                loaded = True

        if not loaded:
            raise FileNotFoundError(
                "No pretrained classifier found. Please provide classifier_path pointing to a PyTorch state_dict"
            )

        self.classifier.eval()

        # state
        self._target_image: t.Optional[np.ndarray] = None
        self._target_label: t.Optional[int] = None
        self._mask: t.Optional[np.ndarray] = None
        self._pos: t.Optional[t.Tuple[int, int]] = None
        self._steps = 0

    def _load_classifier_state(self, path: str) -> None:
        raw = torch.load(path, map_location=self.device)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
        else:
            state = raw
        try:
            alt = SmallCNN(num_classes=10, padding=0).to(self.device)
            alt.load_state_dict(state)
            self.classifier = alt
            return
        except Exception as e:
            raise RuntimeError(f"Failed to load classifier state dict from {path}: {e}")

    def seed(self, seed: t.Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _sample_target(self) -> None:
        if self._mnist is None:
            raise RuntimeError("MNIST dataset not available; cannot sample target image.")
        i = np.random.randint(0, len(self._mnist))
        img, label = self._mnist[i]
        arr = img.squeeze(0).numpy().astype(np.float32)
        self._target_image = arr
        self._target_label = int(label)

    def reset(self, *, seed: t.Optional[int] = None, options: t.Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        self._sample_target()
        self._mask = np.zeros((self.img_h, self.img_w), dtype=bool)
        r = np.random.randint(0, self.img_h)
        c = np.random.randint(0, self.img_w)
        self._pos = (r, c)
        self._mask[r, c] = True
        self._steps = 0
        obs = self._build_observation()
        return obs, {}

    def _build_observation(self) -> np.ndarray:
        assert self._target_image is not None and self._mask is not None and self._pos is not None
        mask = self._mask.astype(np.float32)
        partial = np.where(self._mask, self._target_image, 0.0).astype(np.float32)
        loc = np.zeros_like(mask, dtype=np.float32)
        r, c = self._pos
        loc[r, c] = 1.0
        obs = np.stack([mask, partial, loc], axis=0)
        return obs

    def step(self, action: int):
        assert self._target_image is not None and self._mask is not None and self._pos is not None
        r, c = self._pos
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.img_h - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        elif action == 3:
            c = min(self.img_w - 1, c + 1)
        else:
            raise ValueError(f"Invalid action {action}")
        self._pos = (r, c)
        self._mask[r, c] = True
        self._steps += 1

        obs = self._build_observation()

        # classifier uses partial image only (like original env)
        partial = np.where(self._mask, self._target_image, 0.0).astype(np.float32)
        with torch.no_grad():
            timg = torch.tensor(partial).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.classifier(timg)
            probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
            maxp = float(np.max(probs))
            pred = int(np.argmax(probs))

        info = {"classifier_max_confidence": maxp, "classifier_pred": pred, "true_label": self._target_label}
        terminated = bool(maxp >= self.confidence_threshold)
        truncated = bool((self._steps >= self.max_steps) and not terminated)
        reward = 1.0 if terminated else 0.0

        if terminated or truncated:
            pixels_seen = int(self._mask.sum()) if self._mask is not None else 0
            info["pixels_seen"] = pixels_seen
            info["moves"] = int(self._steps)

        return obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human"):
        if self._target_image is None or self._mask is None or self._pos is None:
            print("Env not initialized")
            return
        if mode == "human":
            disp = self._target_image.copy()
            disp[~self._mask] = -1.0
            r, c = self._pos
            disp[r, c] = 2.0
            np.set_printoptions(precision=2, suppress=True)
            print(disp)
            return
        if mode == "cv2":
            try:
                import cv2
            except Exception as e:
                raise RuntimeError("cv2 not available; install opencv-python") from e
            mask = self._mask.astype(bool)
            img_gray = (np.where(mask, self._target_image, 0.0) * 255.0).clip(0, 255).astype(np.uint8)
            b = np.where(mask, img_gray, 255).astype(np.uint8)
            g = np.where(mask, img_gray, 0).astype(np.uint8)
            rch = np.where(mask, img_gray, 0).astype(np.uint8)
            img_bgr = np.stack([b, g, rch], axis=-1)
            scale = 20
            img_large = cv2.resize(img_bgr, (self.img_w * scale, self.img_h * scale), interpolation=cv2.INTER_NEAREST)
            r0, c0 = self._pos
            center = (int((c0 + 0.5) * scale), int((r0 + 0.5) * scale))
            radius = max(2, scale // 2)
            cv2.circle(img_large, center, radius, (0, 0, 255), -1)
            cv2.imshow("ConvActiveExplorer", img_large)
            cv2.waitKey(1)
            return
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self) -> None:
        return
