"""Evaluate a saved framework model from fresh images, with plasticity off."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .Column import Column
from .data.config import Config as DataConfig
from .data.pipeline import build_transform, make_caltech_loaders, make_mnist_loaders


def evaluate(checkpoint, dataset="caltech", official_test=False, batch_size=128):
    torch.set_num_threads(3)
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(checkpoint)
    model = Column.load(checkpoint)
    config = DataConfig()
    config.paths.caltech_path = str(root / "data/caltech")
    config.paths.mnist_path = str(root / "data/mnist")
    if official_test:
        if dataset != "mnist":
            raise ValueError("Official test mode is for MNIST")
        from torchvision.datasets import MNIST

        ds = MNIST(
            config.paths.mnist_path,
            train=False,
            download=True,
            transform=build_transform(config),
        )
    else:
        factory = make_caltech_loaders if dataset == "caltech" else make_mnist_loaders
        ds = factory(config, seed=42)[1].dataset
    n = len(ds) if dataset == "caltech" or official_test else 5000
    before = model.state_dict()
    predictions, labels = [], []
    for start in range(0, n, batch_size):
        samples = [ds[i] for i in range(start, min(start + batch_size, n))]
        images = torch.stack([x for x, y in samples])
        labels.extend(int(y) for x, y in samples)
        predictions.extend(model.predict(images).cpu().tolist())
        if start % 512 == 0:
            print(f"{dataset}: {min(start + batch_size, n)}/{n}", flush=True)
    after = model.state_dict()
    assert torch.equal(before["kernels"], after["kernels"])
    assert torch.equal(before["wins"], after["wins"])
    assert torch.equal(before["decision"]["w"], after["decision"]["w"])
    assert before["decision"]["reward_baseline"] == after["decision"]["reward_baseline"]
    if before["context"] is not None:
        for key in ("raw_forward", "raw_reverse", "forward", "reverse"):
            assert torch.equal(before["context"][key], after["context"][key])
    report = dict(
        dataset=dataset,
        official_test=official_test,
        n=n,
        accuracy=sum(p == y for p, y in zip(predictions, labels)) / n,
        predictions=predictions,
        labels=labels,
        weights_unchanged=True,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "column").rglob("*.py")
        },
    )
    original = checkpoint.parent / "metrics.json"
    if original.exists():
        prior = json.loads(original.read_text())
        if "predictions" in prior:
            report["matches_recorded_predictions"] = predictions == prior["predictions"]
    (checkpoint.parent / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(
        {
            k: v
            for k, v in report.items()
            if k not in ("predictions", "labels", "source_sha256")
        },
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--dataset", default="caltech", choices=["caltech", "mnist"])
    p.add_argument("--official-test", action="store_true")
    p.add_argument("--batch-size", type=int, default=128)
    evaluate(**vars(p.parse_args()))
