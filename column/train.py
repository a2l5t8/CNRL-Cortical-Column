"""Fresh end-to-end training. Every plastic stage stays enabled by default."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .Column import Column, Config
from .data.config import Config as DataConfig
from .data.pipeline import make_caltech_loaders, make_mnist_loaders


def load_data(dataset, train_limit=None, val_limit=None):
    root = Path(__file__).resolve().parents[1]
    cfg = DataConfig()
    cfg.paths.caltech_path = str(root / "data/caltech")
    cfg.paths.mnist_path = str(root / "data/mnist")
    factory = make_caltech_loaders if dataset == "caltech" else make_mnist_loaders
    loaders = factory(cfg, seed=42)
    limits = (
        train_limit or (320 if dataset == "caltech" else 20000),
        val_limit or (80 if dataset == "caltech" else 5000),
    )
    data = []
    for loader, limit in zip(loaders, limits):
        samples = [loader.dataset[i] for i in range(min(limit, len(loader.dataset)))]
        data.append(
            (
                torch.stack([x for x, y in samples]),
                torch.tensor([int(y) for x, y in samples]),
            )
        )
    return data


def dataset_config(dataset, seed, maps, pool, kernel, **overrides):
    return Config(
        classes=2 if dataset == "caltech" else 10,
        maps=maps or (25 if dataset == "caltech" else 100),
        pool=pool,
        kernel_size=kernel,
        winners=5 if dataset == "caltech" else 16,
        sensory_steps=32 if dataset == "caltech" else 48,
        learning_rate=0.003 if dataset == "caltech" else 0.001,
        noise=0.003 if dataset == "caltech" else 0.01,
        seed=seed,
        **overrides,
    )


def run(
    dataset="caltech",
    epochs=20,
    seed=42,
    maps=None,
    pool=5,
    kernel=11,
    output="outputs/framework_full",
    train_limit=None,
    val_limit=None,
):
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    data = load_data(dataset, train_limit, val_limit)
    config = dataset_config(dataset, seed, maps, pool, kernel)
    root = Path(__file__).resolve().parents[1]
    model = Column(config)
    model.initialize()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "column").rglob("*.py")
    }
    start = time.time()
    report = {}

    def save_epoch(model, metrics):
        nonlocal report
        predictions = model.validation_predictions
        report = dict(
            dataset=dataset,
            config=asdict(config),
            n_train=len(data[0][1]),
            n_validation=len(predictions),
            history=model.history,
            final_accuracy=metrics["validation_accuracy"],
            predictions=predictions.tolist(),
            labels=data[1][1].tolist(),
            seconds=time.time() - start,
            complete=metrics["epoch"] == epochs,
            source_sha256=source_hashes,
            frame_anchor=[0, 0],
            plasticity=["l4", "bindings", "decision"],
        )
        (output / "metrics.json").write_text(json.dumps(report, indent=2))
        model.save(output / "model.pt")
        print(json.dumps(metrics), flush=True)

    model.train(
        *data[0],
        epochs=epochs,
        batch_size=32,
        validation_data=data[1],
        callback=save_epoch,
    )
    return model, report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="caltech", choices=["caltech", "mnist"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--maps", type=int)
    p.add_argument("--pool", type=int, default=5)
    p.add_argument("--kernel", type=int, default=11)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--val-limit", type=int)
    p.add_argument("--output", default="outputs/framework_full")
    run(**vars(p.parse_args()))
