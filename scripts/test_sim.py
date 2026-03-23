"""
test_sim.py — Verify the STARC sim scene loads and both cameras render.

Usage:
    cd /home/mecha/starc
    python scripts/test_sim.py

Outputs:
    outputs/overhead_cam.png   — top-down view of workspace
    outputs/wrist_cam.png      — gripper-eye view
"""

import os
import pathlib

import mujoco
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

SCENE_PATH = pathlib.Path(__file__).parent.parent / "sim" / "scene.xml"
OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CAMERAS = ["overhead_cam", "wrist_cam"]
RENDER_W, RENDER_H = 640, 480


def main():
    # Load model
    print(f"Loading: {SCENE_PATH}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    print(f"Model: {model.nq} DOF, {model.nbody} bodies")

    # List joints
    joints = [model.joint(i).name for i in range(model.njnt)]
    print(f"Joints ({model.njnt}): {', '.join(joints)}")

    # List cameras
    cameras = [model.camera(i).name for i in range(model.ncam)]
    print(f"Cameras ({model.ncam}): {', '.join(cameras)}")

    for cam in CAMERAS:
        if cam not in cameras:
            print(f"  WARNING: camera '{cam}' not found in model")

    # Step simulation to settle physics
    mujoco.mj_resetData(model, data)
    for _ in range(100):
        mujoco.mj_step(model, data)
    print("Sim: 100 steps completed")

    # Render each camera
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    for cam_name in CAMERAS:
        if cam_name not in cameras:
            continue
        renderer.update_scene(data, camera=cam_name)
        pixels = renderer.render()  # shape: (H, W, 3), uint8

        out_path = OUTPUT_DIR / f"{cam_name}.png"
        if HAS_PIL:
            Image.fromarray(pixels).save(out_path)
            print(f"Rendered {cam_name} → {out_path}")
        else:
            # Fallback: save raw bytes as PPM (viewable without PIL)
            out_path = out_path.with_suffix(".ppm")
            with open(out_path, "wb") as f:
                f.write(f"P6\n{RENDER_W} {RENDER_H}\n255\n".encode())
                f.write(pixels.tobytes())
            print(f"Rendered {cam_name} → {out_path}  (PPM, no Pillow installed)")

    renderer.close()
    print("\nAll done. Check outputs/ for rendered images.")


if __name__ == "__main__":
    main()
