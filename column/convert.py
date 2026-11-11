"""Explicit migration of a preserved checkpoint; never described as scratch training."""

import argparse
import json
from pathlib import Path

import torch

from .Column import Column, Config
from .data.config import Config as DataConfig
from .data.pipeline import build_transform, make_caltech_loaders, make_mnist_loaders


def convert(dataset, output, legacy_root=None):
    root = Path(__file__).resolve().parents[1]
    legacy = (
        Path(legacy_root) if legacy_root else root.parent / "claude_retired_2026-11-09"
    )
    old = torch.load(
        legacy
        / f"experiments/spike_count_rstdp/results/{dataset}_simplified_column.pt",
        weights_only=False,
        map_location="cpu",
    )
    kernels = torch.load(
        legacy / f"outputs/l4_kernel_study/{dataset}/l4_best_weights.pt",
        weights_only=True,
        map_location="cpu",
    ).float()
    d = old["decision"]
    config = Config(
        classes=d["C"],
        maps=len(kernels),
        kernel_size=kernels.shape[-1],
        pool=5,
        winners=d["k"],
        sensory_steps=32 if dataset == "caltech" else 48,
        decision_steps=d["steps"],
        learning_rate=d["lr"],
        noise=d["noise"],
        decision_gain=d["gain"],
    )
    model = Column(config)
    model.initialize()
    model.l4.conv.raw_weights.copy_(kernels.to(config.device))
    model.l4.stdp.scale(model.l4.conv)
    context = old["context"]
    model.l23.apical.raw = context["raw_forward"].to(config.device).clone()
    model.l56.apical.raw = context["raw_reverse"].to(config.device).clone()
    # Preserve the stored efficacies exactly, not a newly recalibrated estimate.
    model.l23.apical.weights = (
        context["binding"].reshape(-1, d["F"]).to(config.device).clone()
    )
    model.l56.apical.weights = context["reverse_binding"].to(config.device).clone()
    model.decision.w = d["w"].to(config.device).clone()
    model.net.baseline = d["reward_baseline"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.pt")
    return model


def evaluate(dataset, output, official_test=False, legacy_root=None):
    torch.set_num_threads(3)
    model = convert(dataset, output, legacy_root)
    root = Path(__file__).resolve().parents[1]
    config = DataConfig()
    config.paths.caltech_path = str(root / "data/caltech")
    config.paths.mnist_path = str(root / "data/mnist")
    factory = make_caltech_loaders if dataset == "caltech" else make_mnist_loaders
    ds = factory(config, seed=42)[1].dataset
    if official_test:
        from torchvision.datasets import MNIST

        ds = MNIST(
            config.paths.mnist_path,
            train=False,
            download=True,
            transform=build_transform(config),
        )
    n = len(ds) if dataset == "caltech" or official_test else 5000
    predictions, labels = [], []
    batch = 128
    for start in range(0, n, batch):
        samples = [ds[i] for i in range(start, min(start + batch, n))]
        images = torch.stack([x for x, y in samples])
        labels.extend(int(y) for x, y in samples)
        predictions.extend(model.predict(images).cpu().tolist())
        if start % 512 == 0:
            print(f"{dataset}: {min(start + batch, n)}/{n}", flush=True)
    report = dict(
        dataset=dataset,
        official_test=official_test,
        n=n,
        accuracy=sum(p == y for p, y in zip(predictions, labels)) / n,
        labels=labels,
        predictions=predictions,
        provenance="Converted preserved weights; fresh retinal simulation; no retraining",
    )
    Path(output, "metrics.json").write_text(json.dumps(report, indent=2))
    print(
        {k: v for k, v in report.items() if k not in ("labels", "predictions")},
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="caltech", choices=["caltech", "mnist"])
    p.add_argument("--output", default="outputs/framework_converted")
    p.add_argument("--official-test", action="store_true")
    p.add_argument("--legacy-root", default=None)
    evaluate(**vars(p.parse_args()))
