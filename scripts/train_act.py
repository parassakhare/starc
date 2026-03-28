"""
Train ACT policy on collected cube transfer dataset.

Local validation mode (default):
    python scripts/train_act.py --mode local

Full GPU training (RunPod):
    python scripts/train_act.py --mode full --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def get_policy_features(dataset_features: dict) -> tuple[dict, dict]:
    """Convert dataset features to policy input/output feature dicts."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.utils import dataset_to_policy_features

    features = dataset_to_policy_features(dataset_features)
    input_features = {}
    output_features = {}
    for key, ft in features.items():
        if ft.type is FeatureType.ACTION:
            output_features[key] = ft
        else:
            input_features[key] = ft
    return input_features, output_features


def create_act_config(
    input_features: dict,
    output_features: dict,
    chunk_size: int = 50,
    dim_model: int = 256,
    n_heads: int = 8,
    n_encoder_layers: int = 4,
    n_decoder_layers: int = 1,
    lr: float = 1e-4,
) -> ACTConfig:
    """Create ACT config sized for our task."""
    config = ACTConfig(
        n_obs_steps=1,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        input_features=input_features,
        output_features=output_features,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        dim_model=dim_model,
        n_heads=n_heads,
        dim_feedforward=dim_model * 4,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        dropout=0.1,
        kl_weight=10.0,
        optimizer_lr=lr,
        optimizer_weight_decay=1e-4,
    )
    return config


def train(
    dataset_path: str = "data/starc_cube_transfer",
    repo_id: str = "starc/cube_transfer_scripted",
    output_dir: str = "outputs/act_cube_transfer",
    mode: str = "local",
    device: str = "cpu",
    batch_size: int | None = None,
    max_steps: int | None = None,
    chunk_size: int = 50,
    lr: float = 1e-4,
    save_interval: int | None = None,
):
    # Mode defaults
    if mode == "local":
        batch_size = batch_size or 4
        max_steps = max_steps or 1000
        save_interval = save_interval or 500
        dim_model = 128
    else:  # full
        batch_size = batch_size or 32
        max_steps = max_steps or 100_000
        save_interval = save_interval or 10_000
        dim_model = 256

    print(f"Mode: {mode} | Device: {device} | Batch: {batch_size} | Steps: {max_steps}")

    # Create policy config first (need it for delta_timestamps)
    print("Loading dataset metadata...")
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_path)

    print("Creating ACT policy...")
    input_features, output_features = get_policy_features(ds_meta.features)
    config = create_act_config(
        input_features=input_features,
        output_features=output_features,
        chunk_size=chunk_size,
        dim_model=dim_model,
        lr=lr,
    )

    # Build delta_timestamps from policy config (needed for action chunking)
    from lerobot.datasets.factory import resolve_delta_timestamps
    delta_timestamps = resolve_delta_timestamps(config, ds_meta)
    print(f"  delta_timestamps keys: {list(delta_timestamps.keys()) if delta_timestamps else None}")

    # Load dataset with delta_timestamps
    print("Loading dataset...")
    dataset = LeRobotDataset(repo_id, root=dataset_path, delta_timestamps=delta_timestamps)
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  Frames: {dataset.num_frames}")
    print(f"  Features: {list(dataset.features.keys())}")

    # Verify shapes
    sample = dataset[0]
    for k, v in sample.items():
        if hasattr(v, 'shape') and v.ndim > 0:
            print(f"  {k}: {v.shape}")

    policy = ACTPolicy(config, dataset_stats=dataset.meta.stats)
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if device == "cuda" else 0,
        pin_memory=device == "cuda",
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=lr, weight_decay=1e-4,
    )

    # Training loop
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    policy.train()
    step = 0
    loss_history = []
    t0 = time.time()
    initial_loss = None

    print(f"\nTraining for {max_steps} steps...")
    while step < max_steps:
        for batch in dataloader:
            if step >= max_steps:
                break

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward — returns (loss_tensor, loss_dict)
            loss, loss_info = policy.forward(batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if initial_loss is None:
                initial_loss = loss_val

            step += 1

            # Log
            if step % 100 == 0 or step == 1:
                elapsed = time.time() - t0
                avg_loss = sum(loss_history[-100:]) / len(loss_history[-100:])
                print(
                    f"  step {step:6d}/{max_steps}  "
                    f"loss={loss_val:.4f}  avg100={avg_loss:.4f}  "
                    f"time={elapsed:.0f}s"
                )

            # Save checkpoint
            if save_interval and step % save_interval == 0:
                ckpt_dir = output_path / f"checkpoint-{step:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(str(ckpt_dir))
                print(f"  -> Saved checkpoint to {ckpt_dir}")

    # Save final
    final_dir = output_path / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(final_dir))

    elapsed = time.time() - t0
    final_loss = sum(loss_history[-100:]) / len(loss_history[-100:]) if loss_history else 0

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Steps        : {step}")
    print(f"  Initial loss : {initial_loss:.4f}")
    print(f"  Final loss   : {final_loss:.4f}")
    print(f"  Time         : {elapsed:.1f}s")
    print(f"  Checkpoint   : {final_dir}")

    if mode == "local":
        if final_loss < initial_loss:
            print(f"\n  Pipeline VALIDATED: loss decreased {initial_loss:.4f} -> {final_loss:.4f}")
        else:
            print(f"\n  WARNING: loss did not decrease. Check configuration.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACT policy on cube transfer dataset")
    parser.add_argument("--dataset_path", type=str, default="data/starc_cube_transfer")
    parser.add_argument("--repo_id", type=str, default="starc/cube_transfer_scripted")
    parser.add_argument("--output_dir", type=str, default="outputs/act_cube_transfer")
    parser.add_argument("--mode", type=str, choices=["local", "full"], default="local")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=None)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset_path,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        mode=args.mode,
        device=args.device,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        chunk_size=args.chunk_size,
        lr=args.lr,
        save_interval=args.save_interval,
    )
