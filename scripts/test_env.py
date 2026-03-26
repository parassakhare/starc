"""Smoke test for StarcCubeTransferEnv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sim  # registers starc/CubeTransfer-v0
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

env = gym.make("starc/CubeTransfer-v0")

print("--- reset ---")
obs, info = env.reset(seed=42)
print(f"  obs keys    : {list(obs.keys())}")
print(f"  top shape   : {obs['top'].shape}  dtype={obs['top'].dtype}")
print(f"  agent_pos   : {obs['agent_pos'].round(4)}")
print(f"  info        : {info}")

print("\n--- 10 random steps ---")
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  step {i+1:2d}  reward={reward:.4f}  terminated={terminated}  dist={info['cube_dist_to_target']:.4f}")
    if terminated or truncated:
        break

env.close()

print("\n--- gymnasium env_checker ---")
raw_env = gym.make("starc/CubeTransfer-v0").unwrapped
check_env(raw_env, warn=True)
raw_env.close()

print("\nAll checks passed.")
