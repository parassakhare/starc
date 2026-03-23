# STARC — Skill Training for Autonomous Robotic Capabilities

**Simulation-first, modular VLA pipeline for bimanual robotic manipulation.**

STARC takes a different approach to Physical AI: instead of rushing to hardware,
you build and test every component in simulation first. Vision, Language, and
Action are independently testable modules. The real arm is a deployment target —
not your dev environment.

## Architecture

```
Vision Module  →  Language Module  →  Action Module
(what does       (what task am I     (how do I move
 the scene        being asked         my joints to
 look like?)      to do?)             do it?)
       ↕                ↕                  ↕
   Test each independently in simulation (MuJoCo)
       ↕                ↕                  ↕
              Deploy to real hardware
```

## Quick start (Day 1 — no hardware needed)

```bash
conda create -y -n starc python=3.12 && conda activate starc
pip install lerobot[feetech] mujoco gymnasium

# Train ACT on existing bimanual sim dataset
lerobot-train --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --output_dir=outputs/act_v1 --policy.device=cpu --training.steps=5000
```

## SKILL.md files for Claude Code

Six modular skills, one per pipeline stage:

1. **sim-environment** — MuJoCo, gym-aloha, robosuite, LIBERO
2. **vision-module** — Encoder testing, feature analysis (the V)
3. **language-module** — Task embeddings, VLM integration (the L)
4. **action-module** — ACT, Diffusion Policy, training (the A)
5. **deploy-real** — Sim-to-real, hardware setup (final step)
6. **debug-pipeline** — End-to-end diagnostics

## License

MIT
