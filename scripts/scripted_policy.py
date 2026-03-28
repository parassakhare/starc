"""
Scripted pick-and-place policy for SO-ARM101 cube transfer task.

With the arm tilted 40° forward at z=0.25, it can reach down to the cube.
Uses closed-loop Jacobian IK at each step to drive the gripperframe site
toward Cartesian waypoints.

Run standalone to test:
    python scripts/scripted_policy.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim.starc_env import _CTRL_LOW, _CTRL_HIGH, _CUBE_TARGET

GRIPPER_OPEN = -0.3   # ctrl≈0.5
GRIPPER_CLOSE = -1.0  # ctrl≈-0.17


def normalize_action(joint_targets: np.ndarray) -> np.ndarray:
    """Physical joint positions → normalized [-1, 1]."""
    return 2.0 * (joint_targets - _CTRL_LOW) / (_CTRL_HIGH - _CTRL_LOW) - 1.0


def denormalize_action(action: np.ndarray) -> np.ndarray:
    """Normalized [-1, 1] → physical joint positions."""
    return _CTRL_LOW + (action + 1.0) / 2.0 * (_CTRL_HIGH - _CTRL_LOW)


class ClosedLoopPolicy:
    """Closed-loop pick-and-place using Jacobian IK at each step.

    Phases:
    1. pre-grasp — above cube
    2. descend   — lower to cube height
    3. grasp     — close gripper
    4. lift      — raise with cube
    5. transit   — move to target
    6. lower     — descend at target
    7. release   — open gripper
    8. retreat   — back up
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cube_pos: np.ndarray,
        target_pos: np.ndarray = _CUBE_TARGET,
        grasp_z: float = 0.025,
        pre_grasp_z: float = 0.15,
        pos_threshold: float = 0.015,
        gripper_wait: int = 40,
        damping: float = 1e-4,
        gain: float = 2.0,
    ):
        self.model = model
        self.pos_threshold = pos_threshold
        self.gripper_wait = gripper_wait
        self.damping = damping
        self.gain = gain

        cx, cy = float(cube_pos[0]), float(cube_pos[1])
        tx, ty = float(target_pos[0]), float(target_pos[1])

        self.phases = [
            (np.array([cx, cy, pre_grasp_z]), GRIPPER_OPEN,  "pre-grasp"),
            (np.array([cx, cy, grasp_z]),     GRIPPER_OPEN,  "descend"),
            (None,                            GRIPPER_CLOSE, "grasp"),
            (np.array([cx, cy, pre_grasp_z]), GRIPPER_CLOSE, "lift"),
            (np.array([tx, ty, pre_grasp_z]), GRIPPER_CLOSE, "transit"),
            (np.array([tx, ty, grasp_z]),     GRIPPER_CLOSE, "lower"),
            (None,                            GRIPPER_OPEN,  "release"),
            (np.array([tx, ty, pre_grasp_z]), GRIPPER_OPEN,  "retreat"),
        ]

        self._phase_idx = 0
        self._gripper_counter = 0
        self._converge_counter = 0  # require N consecutive steps below threshold
        self._converge_required = 3
        self._jacp = np.zeros((3, model.nv))
        self._jnt_low = np.array([model.jnt_range[i][0] for i in range(6)])
        self._jnt_high = np.array([model.jnt_range[i][1] for i in range(6)])
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

    @property
    def done(self) -> bool:
        return self._phase_idx >= len(self.phases)

    @property
    def phase_name(self) -> str:
        if self.done:
            return "done"
        return self.phases[self._phase_idx][2]

    def get_action(self, data: mujoco.MjData) -> np.ndarray:
        """Compute next action from current MuJoCo state."""
        if self.done:
            action = normalize_action(data.qpos[:6].copy())
            action[5] = GRIPPER_OPEN
            return np.clip(action, -1.0, 1.0).astype(np.float32)

        cart_target, gripper, label = self.phases[self._phase_idx]

        # Gripper-only phase
        if cart_target is None:
            self._gripper_counter += 1
            if self._gripper_counter >= self.gripper_wait:
                self._gripper_counter = 0
                self._phase_idx += 1
            action = normalize_action(data.qpos[:6].copy())
            action[5] = gripper
            return np.clip(action, -1.0, 1.0).astype(np.float32)

        # Cartesian phase with IK
        site_pos = data.site_xpos[self._site_id].copy()
        dx = cart_target - site_pos
        pos_err = np.linalg.norm(dx)

        if pos_err < self.pos_threshold:
            self._converge_counter += 1
            if self._converge_counter >= self._converge_required:
                self._converge_counter = 0
                self._phase_idx += 1
                action = normalize_action(data.qpos[:6].copy())
                action[5] = gripper
                return np.clip(action, -1.0, 1.0).astype(np.float32)
        else:
            self._converge_counter = 0

        mujoco.mj_jacSite(self.model, data, self._jacp, None, self._site_id)
        J = self._jacp[:, :5]  # arm joints only
        dq = J.T @ np.linalg.solve(
            J @ J.T + self.damping**2 * np.eye(3),
            self.gain * dx,
        )

        arm_targets = data.qpos[:6].copy()
        arm_targets[:5] = np.clip(
            arm_targets[:5] + dq, self._jnt_low[:5], self._jnt_high[:5],
        )
        action = normalize_action(arm_targets)
        action[5] = gripper
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def add_action_noise(
    actions: list[np.ndarray], std: float = 0.01, seed: int | None = None,
) -> list[np.ndarray]:
    """Add Gaussian noise to actions for trajectory diversity."""
    rng = np.random.default_rng(seed)
    return [
        np.clip(a + rng.normal(0, std, size=a.shape).astype(np.float32), -1.0, 1.0)
        for a in actions
    ]


def run_episode(env, noise_std=0.0, seed=None):
    """Run one episode. Returns (actions, obs_list, success, info)."""
    uw = env.unwrapped
    cube_pos = uw.data.qpos[uw._cube_qpos_start: uw._cube_qpos_start + 3].copy()
    policy = ClosedLoopPolicy(uw.model, cube_pos)

    actions, obs_list = [], []
    rng = np.random.default_rng(seed) if noise_std > 0 else None
    info = {}

    for step in range(500):
        obs = uw._get_obs()
        action = policy.get_action(uw.data)
        if rng is not None:
            action = np.clip(
                action + rng.normal(0, noise_std, size=action.shape).astype(np.float32),
                -1.0, 1.0,
            )
        obs_list.append(obs)
        actions.append(action)
        _, _, terminated, truncated, info = env.step(action)
        if policy.done or terminated or truncated:
            break

    return actions, obs_list, info.get("is_success", False), info


# --- Standalone test ---
if __name__ == "__main__":
    import sim
    import gymnasium as gym

    env = gym.make("starc/CubeTransfer-v0")
    successes = 0
    n_test = 10

    for s in range(n_test):
        obs, info = env.reset(seed=s)
        uw = env.unwrapped
        cube_pos = uw.data.qpos[uw._cube_qpos_start: uw._cube_qpos_start + 3].copy()
        site_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        policy = ClosedLoopPolicy(uw.model, cube_pos)
        for step in range(500):
            action = policy.get_action(uw.data)
            obs, reward, terminated, truncated, info = env.step(action)
            if policy.done or terminated or truncated:
                break

        dist = info["cube_dist_to_target"]
        success = info.get("is_success", False)
        if success:
            successes += 1
        cube_final = uw.data.qpos[uw._cube_qpos_start: uw._cube_qpos_start + 3]
        print(f"  seed={s:2d}  start=({cube_pos[0]:.3f},{cube_pos[1]:.3f})  end=({cube_final[0]:.3f},{cube_final[1]:.3f})  dist={dist:.4f}  phase={policy.phase_name}  steps={step+1}  {'OK' if success else 'FAIL'}")

    print(f"\nSuccess: {successes}/{n_test} = {successes/n_test*100:.0f}%")
    env.close()
