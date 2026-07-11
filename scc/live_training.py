"""Small raw-image online-training runner shared by the four live scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Tuple

import torch

from scc.config import Config
from scc.data.pipeline import make_caltech_loaders, make_mnist_loaders
from scc.fully_spiking import FullySpikingConfig, FullySpikingCorticalColumn
from scc.utils import seed_everything


CENTERS: Sequence[Tuple[int, int]] = (
    (20, 20),
    (20, 40),
    (20, 60),
    (40, 20),
    (40, 40),
    (40, 60),
    (60, 20),
    (60, 40),
    (60, 60),
)


def _model_config(dataset: str, seed: int) -> FullySpikingConfig:
    if dataset == "caltech":
        return FullySpikingConfig(
            l23_k=5,
            sensory_steps=32,
            l4_release_steps=32,
            decision_steps=8,
            decision_synaptic_gain=0.3,
            decision_tonic_current=0.5,
            decision_learning_rate=0.02,
            decision_margin=2.0,
            eligibility_tau_pre=0.1,
            apical_learning_rate=0.002,
            apical_gain=0.01,
            random_state=seed,
        )
    return FullySpikingConfig(
        l23_k=16,
        sensory_steps=48,
        l4_release_steps=48,
        decision_steps=8,
        decision_synaptic_gain=0.3,
        decision_tonic_current=0.5,
        decision_learning_rate=0.05,
        decision_margin=6.0,
        eligibility_tau_pre=0.1,
        apical_learning_rate=0.002,
        apical_gain=0.01,
        random_state=seed,
    )


def _dataset(dataset: str, batch_size: int, seed: int):
    if dataset == "caltech":
        cfg = Config.caltech()
        train_loader, val_loader = make_caltech_loaders(
            cfg,
            batch_size=batch_size,
            seed=seed,
        )
        return cfg, train_loader, val_loader, 2, 25
    cfg = Config.mnist()
    train_loader, val_loader = make_mnist_loaders(
        cfg,
        batch_size=batch_size,
        seed=seed,
    )
    return cfg, train_loader, val_loader, 10, 100


def _initial_kernels(
    dataset: str,
    n_features: int,
    seed: int,
    train_l4: bool,
) -> torch.Tensor:
    if not train_l4:
        path = Path("outputs/l4_kernel_study") / dataset / "l4_best_weights.pt"
        return torch.load(path, map_location="cpu", weights_only=False).float()
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n_features, 1, 11, 11, generator=generator)


def run_live_training(
    *,
    dataset: str,
    train_l4: bool,
    output_dir: Path,
    device: str,
    seed: int,
    batch_size: int,
    train_samples: int,
    eval_samples: int,
    epochs: int,
    binding_epochs: int,
    l4_samples: int,
    l4_epochs: int,
) -> dict:
    """Run raw-image L4/cortical learning without an event cache."""
    seed_everything(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, train_loader, val_loader, n_classes, n_features = _dataset(
        dataset,
        batch_size,
        seed,
    )
    kernels = _initial_kernels(dataset, n_features, seed, train_l4)
    model = FullySpikingCorticalColumn(
        n_classes=n_classes,
        n_locations=len(CENTERS),
        l4_kernels=kernels,
        cfg=_model_config(dataset, seed),
        device=device,
    )

    if train_l4:
        print(f"[{dataset}] online L4 STDP from {l4_samples} raw images")
        model.fit_l4_images(
            train_loader,
            CENTERS,
            epochs=l4_epochs,
            n_samples=l4_samples,
        )
        torch.save(
            model.sync_l4_kernels(),
            output_dir / "learned_l4_weights.pt",
        )

    print(
        f"[{dataset}] raw-image cortical learning: {epochs} epochs, "
        f"{train_samples} images per epoch"
    )
    model.fit_joint_images(
        train_loader,
        CENTERS,
        epochs=epochs,
        n_samples=train_samples,
        binding_epochs=binding_epochs,
        binding_update_steps=1,
    )
    evaluation = model.evaluate_images(
        val_loader,
        CENTERS,
        n_samples=eval_samples,
    )

    checkpoint_path = output_dir / "best_fully_spiking_column.pt"
    metadata = {
        "dataset": dataset,
        "seed": seed,
        "centers": list(CENTERS),
        "training_mode": "raw_image_online",
        "l4_mode": "online_stdp_then_frozen" if train_l4 else "pretrained_frozen",
        "event_cache_used": False,
        "event_cache_created": False,
        "learning_replays": False,
        "forced_output_spikes": False,
        "raw_image_accuracy": evaluation["accuracy"],
    }
    torch.save(model.checkpoint(metadata), checkpoint_path)
    report = {
        "dataset": dataset,
        "device": device,
        "backend": model.backend,
        "raw_image_online": True,
        "event_cache_used": False,
        "event_cache_created": False,
        "l4_mode": metadata["l4_mode"],
        "train_samples_per_epoch": train_samples,
        "epochs": epochs,
        "binding_epochs": binding_epochs,
        "l4_training_history": model.l4_training_history,
        "cortical_training_history": model.training_history,
        "raw_image_validation_accuracy": evaluation["accuracy"],
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "live_training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return report


def run_cli(dataset: str, train_l4: bool, default_output: str) -> None:
    defaults = {
        "caltech": {
            "train_samples": 320,
            "eval_samples": 80,
            "epochs": 3,
            "binding_epochs": 3,
        },
        "mnist": {
            "train_samples": 20000,
            "eval_samples": 5000,
            "epochs": 10,
            "binding_epochs": 8,
        },
    }[dataset]
    parser = argparse.ArgumentParser(
        description=(
            f"Raw-image online training for {dataset}; no cortical event cache."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path(default_output))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=defaults["train_samples"],
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=defaults["eval_samples"],
    )
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument(
        "--binding-epochs",
        type=int,
        default=defaults["binding_epochs"],
    )
    parser.add_argument("--l4-samples", type=int, default=200)
    parser.add_argument("--l4-epochs", type=int, default=1)
    args = parser.parse_args()
    run_live_training(
        dataset=dataset,
        train_l4=train_l4,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        epochs=args.epochs,
        binding_epochs=args.binding_epochs,
        l4_samples=args.l4_samples,
        l4_epochs=args.l4_epochs,
    )
