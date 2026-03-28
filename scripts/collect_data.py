"""
Collect scripted pick-and-place episodes and save as a LeRobot dataset.

Usage:
    python scripts/collect_data.py --num_episodes 200
    python scripts/collect_data.py --num_episodes 5 --output_dir data/test  # quick test
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import sim  # noqa: F401  — registers starc/CubeTransfer-v0
import gymnasium as gym
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from scripted_policy import ClosedLoopPolicy, normalize_action

REPO_ID = "starc/cube_transfer_scripted"
TASK_DESCRIPTION = "Pick up the red cube and place it at the target location."

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "forearm_roll",
    "wrist",
    "gripper",
]

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": JOINT_NAMES,
    },
    "observation.images.top": {
        "dtype": "image",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": JOINT_NAMES,
    },
}


def collect_dataset(
    num_episodes: int = 200,
    output_dir: str = "data/starc_cube_transfer",
    seed: int = 42,
    noise_std: float = 0.005,
    fps: int = 50,
    max_attempts_per_episode: int = 5,
):
    output_path = Path(output_dir)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
        print(f"Removed existing dataset at {output_path}")

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=fps,
        features=FEATURES,
        root=output_path,
        robot_type="so101",
        use_videos=False,
        image_writer_threads=4,
    )

    env = gym.make("starc/CubeTransfer-v0")
    rng = np.random.default_rng(seed)

    successes = 0
    total_attempts = 0
    total_frames = 0
    episode_lengths = []
    t0 = time.time()

    ep = 0
    attempt_seed = seed
    while ep < num_episodes:
        # Reset environment
        obs, info = env.reset(seed=attempt_seed)
        attempt_seed += 1
        total_attempts += 1

        uw = env.unwrapped
        cube_pos = uw.data.qpos[uw._cube_qpos_start : uw._cube_qpos_start + 3].copy()
        policy = ClosedLoopPolicy(uw.model, cube_pos)

        # Collect episode frames
        frames = []
        for step in range(500):
            obs = uw._get_obs()
            action = policy.get_action(uw.data)

            # Add small noise for trajectory diversity
            if noise_std > 0:
                action = np.clip(
                    action + rng.normal(0, noise_std, size=action.shape).astype(np.float32),
                    -1.0, 1.0,
                )

            frames.append({
                "observation.state": obs["agent_pos"].astype(np.float32),
                "observation.images.top": obs["top"],
                "action": action,
                "task": TASK_DESCRIPTION,
            })

            _, _, terminated, truncated, info = env.step(action)
            if policy.done or terminated or truncated:
                break

        success = info.get("is_success", False)

        if not success:
            # Discard failed episode — don't save to dataset
            continue

        # Save successful episode
        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()

        successes += 1
        ep += 1
        n_frames = len(frames)
        total_frames += n_frames
        episode_lengths.append(n_frames)

        elapsed = time.time() - t0
        rate = ep / elapsed if elapsed > 0 else 0
        dist = info.get("cube_dist_to_target", -1)
        print(
            f"  ep {ep:3d}/{num_episodes}  "
            f"frames={n_frames:3d}  dist={dist:.4f}  "
            f"attempts={total_attempts}  "
            f"rate={rate:.1f} ep/s  "
            f"phase={policy.phase_name}"
        )

    env.close()

    elapsed = time.time() - t0
    success_rate = successes / total_attempts if total_attempts > 0 else 0
    avg_len = np.mean(episode_lengths) if episode_lengths else 0

    print(f"\n{'='*60}")
    print(f"Collection complete!")
    print(f"  Episodes saved : {successes}")
    print(f"  Total attempts : {total_attempts}")
    print(f"  Success rate   : {success_rate:.0%}")
    print(f"  Total frames   : {total_frames}")
    print(f"  Avg ep length  : {avg_len:.0f} frames")
    print(f"  Time           : {elapsed:.1f}s")
    print(f"  Dataset path   : {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect scripted pick-and-place episodes")
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="data/starc_cube_transfer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_std", type=float, default=0.005)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    collect_dataset(
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
        noise_std=args.noise_std,
        fps=args.fps,
    )
