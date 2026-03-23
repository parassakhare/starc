# STARC v2 — Simulation-First Modular VLA Pipeline

## The Philosophy: Simulation IS the Development Platform

The original STARC plan made a mistake: it treated hardware as the starting point
and simulation as an afterthought. Your instinct is correct — flip it entirely.

**The real arm is just a deployment target.** All development, all iteration, all
component testing happens in simulation. When something works in sim, it should
work on hardware with minimal gap. The key to narrowing that gap isn't "hoping
sim transfers" — it's building the sim environment so faithfully that the
transition is boring.

This revised plan restructures STARC around three principles:

1. **Simulation-first**: Load a precise digital twin of the SO-ARM101 bimanual
   setup into MuJoCo. All coding, testing, and iteration happens here.
2. **Modular V-L-A testing**: Vision, Language, and Action are three separate
   pipelines you can independently develop, test, and swap. Not one monolithic
   model you pray works end-to-end.
3. **Rapid prototyping loop**: Change a component, test in sim, see results in
   seconds. No USB cables, no calibration drift, no servo overheating.

---

## Architecture: The Modular VLA Pipeline

Instead of treating the VLA as one black box, STARC v2 decomposes it into
independently testable modules:

```
┌─────────────────────────────────────────────────────────────┐
│                    STARC VLA Pipeline                         │
│                                                               │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐       │
│  │  VISION  │───▶│ LANGUAGE  │───▶│     ACTION       │       │
│  │  Module   │    │  Module    │    │     Module        │       │
│  │           │    │            │    │                    │       │
│  │ Camera →  │    │ "pick up   │    │ ACT / Diffusion / │       │
│  │ Scene     │    │  the red   │    │ Flow Matching      │       │
│  │ Under-    │    │  block"    │    │                    │       │
│  │ standing  │    │     ↓      │    │ joint positions    │       │
│  │           │    │ Task       │    │ → robot moves      │       │
│  │ SigLIP /  │    │ Embedding  │    │                    │       │
│  │ DINOv2 /  │    │            │    │                    │       │
│  │ ResNet18  │    │ SmolLM /   │    │                    │       │
│  └──────────┘    │ Gemma      │    └──────────────────┘       │
│                   └───────────┘                                │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              SIMULATION ENVIRONMENT                       │ │
│  │  MuJoCo + robosuite / gym-aloha / LIBERO                │ │
│  │  Bimanual SO-ARM101 digital twin                         │ │
│  │  RGB cameras (simulated), domain randomization           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           ↕ (when ready)                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              REAL HARDWARE                                │ │
│  │  SO-ARM101 + Logitech C920s + LeRobot                   │ │
│  │  Same action space, same observation format              │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Why modular matters

When an end-to-end VLA fails, you don't know why. Is the vision encoder not
recognizing the object? Is the language module parsing the instruction wrong?
Is the action head generating bad trajectories? With monolithic training, you're
debugging a 450M-parameter black box.

With the modular approach:

- **Test Vision alone**: Feed sim images → does the encoder correctly identify
  object positions, colors, shapes? Measure this with ground-truth sim data.
- **Test Language alone**: Feed task descriptions → does the language module
  produce the right task embedding? Swap between "pick up the red cube" and
  "grab the crimson block" — does the embedding remain similar?
- **Test Action alone**: Given perfect state information (from sim ground truth),
  does the action policy complete the task? If yes, your action module works;
  failures must come from upstream.

This is how you iterate fast: isolate the bottleneck, fix it, move on.

---

## Simulation Stack (runs entirely on i7-1335U)

### Layer 1: Physics — MuJoCo (CPU, real-time)

MuJoCo is the physics backbone. It runs at 1000+ Hz on your CPU for a dual-arm
setup — faster than real-time. The SO-ARM100 has an official model in MuJoCo
Menagerie, which you'll use as the base.

```bash
pip install mujoco gymnasium
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

### Layer 2: Task Framework — robosuite + gym-aloha

You have two strong options that run on CPU:

**robosuite v1.5** (Stanford/ARISE Initiative):
- Built on MuJoCo, provides bimanual manipulation tasks out of the box
  (peg-in-hole, lift, handover, transport)
- Modular: swap robot models, grippers, arenas, objects programmatically
- Built-in domain randomization, RGB-D camera simulation, teleoperation
- Gym-compatible API (reset/step)
- Runs on CPU, no GPU required for physics or basic rendering

**gym-aloha** (LeRobot native):
- ALOHA bimanual sim with insertion and cube transfer tasks
- Direct LeRobot integration (same data format, same training scripts)
- Less flexible for custom tasks but zero-friction LeRobot compatibility

**Recommendation**: Start with **gym-aloha** for immediate LeRobot compatibility
(Days 1–4), then build custom tasks in **robosuite** for more complex scenarios
(Days 5+). Both use MuJoCo underneath — the physics are identical.

### Layer 3: Benchmarks — LIBERO (the standard VLA evaluation)

LIBERO is the de facto benchmark for VLA evaluation. It provides 130 tasks
across 5 suites (Spatial, Object, Goal, 90 short-horizon, 10 long-horizon).
LeRobot has native LIBERO integration:

```bash
pip install lerobot[libero]
export MUJOCO_GL=egl  # headless rendering

lerobot-eval \
  --policy.path=your_policy \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=10
```

This runs on CPU (slowly, but it works). LIBERO gives you a standardized
success rate to measure progress against published VLA results.

**Critical caveat**: Recent research (LIBERO-PRO, Oct 2025) showed that VLA
models scoring >90% on standard LIBERO often collapse to 0% under minor
perturbations. Models memorize action sequences rather than understanding tasks.
Keep this in mind — high LIBERO scores don't mean your model generalizes.

### Layer 4: Digital Twin Fidelity (narrowing the sim-to-real gap)

The gap narrows when you match these carefully:

| Parameter | Sim value | How to match real |
|-----------|-----------|-------------------|
| Joint DOF | 6 + gripper per arm | Match SO-ARM101 MJCF model exactly |
| Action space | Joint position targets | Same for LeRobot real control |
| Observation space | RGB 640×480 + joint state | Same cameras, same resolution |
| Control frequency | 15–30 Hz | Match LeRobot recording FPS |
| Camera placement | Overhead + front | Mount real cameras at same positions |
| Object properties | Mass, friction, size | Measure and set in XML |
| Lighting | Consistent, diffuse | Control real workspace lighting |

The more of these you nail in sim, the more boring the real deployment becomes.

---

## Revised 2-Week Sprint Plan

### Week 1: Simulation Mastery & Modular Components

**Days 1–2: Simulation environment setup**

Get the full sim stack running on your laptop. No hardware needed.

```bash
# Core install
conda create -y -n starc python=3.12 && conda activate starc
pip install lerobot[feetech] mujoco gymnasium robosuite

# Clone models
git clone https://github.com/google-deepmind/mujoco_menagerie.git

# Verify gym-aloha runs
python -c "
import gymnasium as gym
env = gym.make('gym_aloha/AlohaTransferCube-v0', obs_type='pixels')
obs, info = env.reset()
print(f'Observation keys: {obs.keys()}')
print(f'Action space: {env.action_space}')
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
print('Sim running on CPU — no GPU needed.')
"
```

Train ACT on existing ALOHA sim datasets (these are already on HF Hub):

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --output_dir=outputs/act_aloha_sim_v1 \
  --policy.device=cpu \
  --training.steps=10000
```

Yes, training on CPU will be slow (~10× slower than GPU). But 10k steps is
enough to verify the pipeline works. Do the full 100k run on cloud GPU later.

**Days 3–4: Modular Vision testing**

The Vision module is the most testable in isolation.

```python
# Test vision encoders on sim-rendered images
import torch
from PIL import Image

# Render a frame from sim
env.reset()
obs, _, _, _, _ = env.step(env.action_space.sample())
image = obs['pixels']['top']  # or however your env returns images

# Test 1: ResNet18 (ACT's default encoder)
from torchvision.models import resnet18
from torchvision import transforms
encoder = resnet18(pretrained=True)
encoder.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
features = encoder(transform(Image.fromarray(image)).unsqueeze(0))
print(f"ResNet18 features: {features.shape}")

# Test 2: DINOv2 (used by OpenVLA, better representations)
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval()
# ... same image preprocessing, extract features

# Test 3: SigLIP (used by π0, SmolVLA — vision-language aligned)
# Requires transformers library
from transformers import AutoModel, AutoProcessor
siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224")
```

**What you're measuring**: Do the features differentiate between scenes?
Feed 100 sim frames with the cube at different positions. Cluster the features.
If the encoder can distinguish positions, it's working. If features are
identical regardless of cube position, that encoder won't work for your task.

**Days 5–6: Modular Action testing**

Test action policies in isolation using ground-truth state from sim (no vision).

```bash
# Train ACT with state-only observations (no images)
# This isolates whether the action module can solve the task
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --output_dir=outputs/act_state_only \
  --policy.device=cpu \
  --training.steps=50000

# Evaluate — if this works well, your action module is solid
lerobot-eval \
  --policy.path=outputs/act_state_only/checkpoints/last/pretrained_model \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=50
```

Then train with vision and compare. The delta tells you exactly how much
the vision encoder is helping (or hurting).

Also try Diffusion Policy on the same task:

```bash
lerobot-train \
  --policy.type=diffusion \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  ...
```

**Day 7: Language module integration**

Now add language conditioning. Two approaches:

**Approach A — SmolVLA (integrated VLA, 450M params)**:
```bash
# Fine-tune SmolVLA on ALOHA sim data (needs GPU)
lerobot-train \
  --policy.type=smolvla \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --output_dir=outputs/smolvla_sim_v1 \
  --policy.device=cuda  # on RunPod
```

**Approach B — Modular: frozen VLM + action head** (more hackable):
```python
# Use a small VLM to produce task embeddings
# Then feed embeddings to your already-trained action policy
from transformers import AutoModelForCausalLM, AutoTokenizer

# SmolLM (135M params, runs on CPU)
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

# Encode task description
inputs = tokenizer("pick up the red cube", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    task_embedding = outputs.hidden_states[-1].mean(dim=1)
    
# Feed task_embedding as conditioning to your action policy
# This is the "modular VLA" approach — test each piece independently
```

### Week 2: Advanced Policies, LIBERO, & Real Hardware

**Days 8–9: LIBERO benchmark evaluation**

Run your trained policies against the standard benchmark:

```bash
pip install lerobot[libero]
export MUJOCO_GL=egl

# Evaluate on LIBERO-Spatial (easiest suite)
lerobot-eval \
  --policy.path=outputs/smolvla_sim_v1/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=10
```

Compare your results against published numbers. SmolVLA achieves ~78% on
SO-100 tasks; π0.5 achieves 85%+ on LIBERO after fine-tuning. Your goal
isn't to beat these — it's to understand what fails and why.

**Days 10–11: Custom sim environments & domain randomization**

Build environments that exactly mirror your planned real workspace:

```python
import robosuite as suite

# Bimanual lift task with domain randomization
env = suite.make(
    "TwoArmLift",
    robots=["Panda", "Panda"],  # swap with SO-ARM100 model later
    controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
    has_renderer=False,  # headless for CPU
    has_offscreen_renderer=True,  # for camera obs
    use_camera_obs=True,
    camera_names=["agentview", "birdview"],
    camera_heights=480,
    camera_widths=640,
)

# Add domain randomization
from robosuite.wrappers import DomainRandomizationWrapper
env = DomainRandomizationWrapper(
    env,
    randomize_every_n_steps=1,
    randomize_color=True,
    randomize_camera=True,
    randomize_lighting=True,
    randomize_dynamics=True,
)
```

**Days 12–13: Real hardware integration (finally)**

Now — and only now — bring in the physical arm. By this point:
- Your action policy is proven in sim
- Your vision encoder is validated on sim-rendered images
- Your language module produces good task embeddings
- You know your LIBERO scores

Assembly and calibration (half a day):
```bash
python -m lerobot.find_port
python -m lerobot.calibrate --robot.type=so101_follower ...
python -m lerobot.teleoperate ...
```

Deploy your sim-trained policy directly:
```bash
python -m lerobot.record \
  --policy.path=outputs/act_aloha_sim_v1/checkpoints/last/pretrained_model \
  --policy.device=cpu \
  --robot.type=so101_follower ...
```

**Expect 30–50% success rate** from pure sim transfer. Then record 10–20 real
demonstrations and fine-tune:

```bash
# Fine-tune the sim-trained policy on real data
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=${HF_USER}/starc_real_demos \
  --policy.path=outputs/act_aloha_sim_v1/checkpoints/last/pretrained_model \
  --training.lr=1e-6 \
  --training.steps=10000 \
  --output_dir=outputs/act_finetuned_real
```

**Day 14: Documentation & retrospective**

Finalize all SKILL.md files with real findings. Push to GitHub. Record demo.

---

## GPU Strategy

Your i7-1335U handles:
- All simulation (MuJoCo, gym-aloha, robosuite, LIBERO)
- Vision encoder inference (ResNet18, small DINOv2)
- ACT/Diffusion inference (10–30 Hz)
- Data collection and visualization
- Small language model inference (SmolLM-135M)

Cloud GPU needed for:
- ACT training 100k steps: RTX 3090, 2–4 hrs, ~$1
- Diffusion Policy training: RTX 3090, 6–12 hrs, ~$2
- SmolVLA fine-tuning: A100 40GB, 4–8 hrs, ~$6
- LIBERO evaluation at scale: RTX 3090, 1–2 hrs, ~$0.50

**Total cloud budget: $15–30**

Best option: RunPod ($0.22/hr RTX 3090, $1.10/hr A100). No commitment,
pay per minute. Alternatively, Google Colab free tier works for ACT but
has session limits.

---

## Key Repos & References

### Simulation
- MuJoCo: github.com/google-deepmind/mujoco
- MuJoCo Menagerie (SO-ARM100 model): github.com/google-deepmind/mujoco_menagerie
- robosuite v1.5: github.com/ARISE-Initiative/robosuite
- gym-aloha: included in LeRobot
- LIBERO: huggingface.co/docs/lerobot/libero
- MuJoCo Playground: playground.mujoco.org
- VLA Evaluation Harness: github.com/allenai/vla-evaluation-harness

### VLA Models (by size, smallest first)
- SmolVLA (450M): HuggingFace, runs on RTX 3060
- X-VLA (0.9B): github.com/2toinf/X-VLA — ICLR 2026, cross-embodiment
- π0/π0.5 (~3B): github.com/Physical-Intelligence/openpi
- OpenVLA (7B): github.com/openvla/openvla

### VLA Architecture Research
- Dual-system (System 1 fast + System 2 slow): NVIDIA GR00T N1, Figure Helix
- Early fusion: EF-VLA (ICLR 2025)
- Chain-of-Affordance: CoA-VLA (ICCV 2025)
- RL fine-tuning: SimpleVLA-RL (99% LIBERO with RL post-training)

### Action Policies
- ACT: Action Chunking with Transformers (recommended starting point)
- Diffusion Policy: multi-modal action generation
- Flow Matching: π0's approach, smooth 50 Hz control

### Core Framework
- LeRobot v0.5+: github.com/huggingface/lerobot
- SO-ARM100 hardware: github.com/TheRobotStudio/SO-ARM100
