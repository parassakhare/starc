"""
Evaluate a trained ACT policy on the cube transfer task.

Usage:
    python scripts/eval_act.py --checkpoint outputs/act_cube_transfer/final --num_episodes 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import sim  # noqa: F401
import gymnasium as gym
from lerobot.policies.act.modeling_act import ACTPolicy


def eval_policy(
    checkpoint: str = "outputs/act_cube_transfer/final",
    num_episodes: int = 50,
    seed: int = 0,
    device: str = "cpu",
    render_dir: str | None = None,
):
    print(f"Loading policy from {checkpoint}...")
    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()
    print(f"  Chunk size: {policy.config.chunk_size}")

    env = gym.make("starc/CubeTransfer-v0")

    successes = 0
    distances = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)

        # Reset policy action queue
        policy.reset()

        for step in range(500):
            # Prepare observation for policy
            state = torch.from_numpy(obs["agent_pos"].astype(np.float32)).unsqueeze(0).to(device)
            image = torch.from_numpy(obs["top"].copy()).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0).to(device)

            with torch.inference_mode():
                action = policy.select_action({
                    "observation.state": state,
                    "observation.images.top": image,
                })

            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)

            if terminated or truncated:
                break

        dist = info.get("cube_dist_to_target", 999)
        success = info.get("is_success", False)
        distances.append(dist)
        if success:
            successes += 1

        status = "OK" if success else "FAIL"
        print(f"  ep {ep+1:3d}/{num_episodes}  dist={dist:.4f}  steps={step+1:3d}  {status}")

    env.close()

    success_rate = successes / num_episodes
    avg_dist = np.mean(distances)
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"  Success rate : {successes}/{num_episodes} = {success_rate:.0%}")
    print(f"  Avg distance : {avg_dist:.4f}")
    print(f"  Checkpoint   : {checkpoint}")
    print(f"{'='*60}")

    return success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained ACT policy")
    parser.add_argument("--checkpoint", type=str, default="outputs/act_cube_transfer/final")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    eval_policy(
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        seed=args.seed,
        device=args.device,
    )
