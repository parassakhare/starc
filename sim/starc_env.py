"""
STARC gymnasium environment: SO-ARM101 single-arm cube transfer task.

Observation space:
  "top"       : (480, 640, 3) uint8   — overhead camera
  "agent_pos" : (6,) float64          — arm joint positions (rad)

Action space: Box(-1, 1, shape=(6,), dtype=float32)
  Normalized joint targets mapped to physical ctrlrange.

Reward: -L2 distance from cube to target position (dense, negative).
Termination: cube within 2 cm of target.
"""

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
import gymnasium as gym

_SCENE_XML = Path(__file__).parent / "scene.xml"

# Actuator control ranges from so101.xml (order: shoulder_pan … gripper)
_CTRL_LOW = np.array([-1.91986, -1.74533, -1.69000, -1.65806, -2.74385, -0.17453], dtype=np.float64)
_CTRL_HIGH = np.array([1.91986, 1.74533, 1.69000, 1.65806, 2.84121, 1.74533], dtype=np.float64)

# Cube start and target positions (XYZ, world frame)
_CUBE_START = np.array([0.18, 0.0, 0.022])
_CUBE_TARGET = np.array([0.28, 0.0, 0.022])
_SUCCESS_THRESHOLD = 0.02  # metres

# Physics settle steps on reset
_SETTLE_STEPS = 100


def _denormalize(action: np.ndarray) -> np.ndarray:
    """Map normalized [-1, 1] action to physical joint position targets."""
    return _CTRL_LOW + (action + 1.0) / 2.0 * (_CTRL_HIGH - _CTRL_LOW)


class StarcCubeTransferEnv(gym.Env):
    """SO-ARM101 cube transfer task in MuJoCo."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        max_episode_steps: int = 500,
        cube_pos_noise: float = 0.02,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.cube_pos_noise = cube_pos_noise

        self.model = mujoco.MjModel.from_xml_path(str(_SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        # Indices
        self._cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self._cube_qpos_start = self.model.jnt_qposadr[self._cube_joint_id]  # freejoint → 7 values

        self.observation_space = spaces.Dict(
            {
                "top": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                "agent_pos": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

        self._step_count = 0
        self._np_random = None

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize cube start XY position
        if self.cube_pos_noise > 0:
            noise = self.np_random.uniform(-self.cube_pos_noise, self.cube_pos_noise, size=2)
        else:
            noise = np.zeros(2)

        qstart = self._cube_qpos_start
        self.data.qpos[qstart : qstart + 3] = _CUBE_START + np.array([noise[0], noise[1], 0.0])
        self.data.qpos[qstart + 3 : qstart + 7] = [1.0, 0.0, 0.0, 0.0]  # unit quaternion

        # Settle physics
        for _ in range(_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        obs = self._get_obs()
        info = {"is_success": False}
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.data.ctrl[:] = _denormalize(action)
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        cube_pos = self.data.qpos[self._cube_qpos_start : self._cube_qpos_start + 3]
        dist = float(np.linalg.norm(cube_pos - _CUBE_TARGET))
        reward = -dist

        terminated = dist < _SUCCESS_THRESHOLD
        truncated = self._step_count >= self.max_episode_steps
        info = {"is_success": terminated, "cube_dist_to_target": dist}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self):
        self._renderer.update_scene(self.data, camera="overhead_cam")
        return self._renderer.render()

    # ------------------------------------------------------------------
    def close(self):
        self._renderer.close()

    # ------------------------------------------------------------------
    def _get_obs(self) -> dict:
        self._renderer.update_scene(self.data, camera="overhead_cam")
        top = self._renderer.render().copy()
        agent_pos = self.data.qpos[:6].copy()
        return {"top": top, "agent_pos": agent_pos}
