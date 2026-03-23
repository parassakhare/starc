# STARC — Skill Training for Autonomous Robotic Capabilities

Simulation-first modular VLA pipeline for bimanual robotic manipulation.
Develop and test Vision, Language, and Action components independently in sim,
deploy to real hardware only when proven.

## Philosophy

**The real arm is a deployment target, not the development platform.**
All iteration happens in simulation. Components are tested in isolation.
The sim-to-real gap is narrowed by faithful digital twin construction.

## Hardware

- **Dev machine**: Intel i7-1335U · 16 GB RAM · 256 GB SSD · Ubuntu 22.04
- **Training GPU**: RunPod cloud RTX 3090 / A100 (remote only)
- **Real robot** (deployment only): SO-ARM101 dual-arm, Feetech STS3215 servos
- **Cameras** (deployment only): 2× Logitech C920

## Software Stack

- **Simulation**: MuJoCo (CPU) + gym-aloha + robosuite v1.5 + LIBERO
- **Framework**: LeRobot v0.5+ (conda env `starc`, Python 3.12)
- **Policies**: ACT (primary), Diffusion Policy, SmolVLA
- **Vision encoders**: ResNet18 (ACT default), DINOv2, SigLIP
- **Language models**: SmolLM-135M (CPU), Gemma (GPU)
- **No ROS2** — pure Python throughout

## Key Commands

```bash
conda activate starc

# Sim — run ALOHA bimanual task
python -c "import gymnasium as gym; env = gym.make('gym_aloha/AlohaTransferCube-v0')"

# Sim — train ACT on existing sim dataset
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task=AlohaTransferCube-v0

# Sim — evaluate policy
lerobot-eval --policy.path=outputs/act_v1/.../pretrained_model \
  --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=50

# Sim — LIBERO benchmark
lerobot-eval --policy.path=... --env.type=libero --env.task=libero_spatial

# Real (deployment only) — discover hardware
python -m lerobot.find_port && python -m lerobot.find_cameras opencv
```

## Directory Layout

```
starc/
├── CLAUDE.md              ← you are here
├── .claude/skills/        ← 6 modular skill files
│   ├── sim-environment/   ← MuJoCo, gym-aloha, robosuite, LIBERO
│   ├── vision-module/     ← encoder testing, feature analysis
│   ├── language-module/   ← task embeddings, VLM integration
│   ├── action-module/     ← ACT, Diffusion, flow matching
│   ├── deploy-real/       ← sim-to-real transfer, hardware setup
│   └── debug-pipeline/    ← end-to-end diagnostics
├── sim/                   ← custom MuJoCo scenes, MJCF files
├── scripts/               ← standalone test scripts
├── data/                  ← datasets (sim and real)
├── outputs/               ← training checkpoints
└── configs/               ← training config overrides
```

## Safety

1. Simulation has no safety concerns — iterate freely.
2. When deploying to real hardware: keep e-stop accessible, start at slow speed.
3. USB ports renumber on reboot — always `find_port` before real-hardware sessions.
4. Cloud GPU: set billing alerts, terminate pods when done.
