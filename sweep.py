"""Run independent fixation-grid Ã— kernel experiments with restartable results."""

import argparse
import gc
import hashlib
import itertools
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from baseline.cnn import PatchCNN, fit
from column import Column
from column.L4 import extract_patches, fixation_geometry
from column.train import dataset_config, load_data

MODES = ("spiking_rf", "spiking_no_rf", "cnn_location", "cnn_no_location")


def make_plan(args):
    if min(args.shards, args.epochs, args.batch_size, args.cnn_batch_size) < 1:
        raise ValueError("Shard count, epochs and batch sizes must be positive")
    if not 1 <= args.shard <= args.shards:
        raise ValueError("Shard number must be between 1 and the shard count")
    if any(
        limit is not None and limit < 1 for limit in (args.train_limit, args.val_limit)
    ):
        raise ValueError("Sample limits must be positive")

    jobs = []
    axes = (args.datasets, args.grids, args.kernels, args.modes, args.seeds)
    for dataset, grid, kernel, mode, seed in itertools.product(
        *(sorted(set(a)) for a in axes)
    ):
        crop, _ = fixation_geometry(grid)
        if not 1 <= kernel <= crop:
            raise ValueError(
                f"Kernel {kernel} does not fit the {crop}px crop for grid {grid}x{grid}"
            )
        spiking = mode.startswith("spiking")
        jobs.append(
            dict(
                protocol=1,
                dataset=dataset,
                grid=grid,
                kernel=kernel,
                crop=crop,
                mode=mode,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size if spiking else args.cnn_batch_size,
                train_limit=args.train_limit,
                val_limit=args.val_limit,
            )
        )
    return jobs


def partition(jobs, count):
    # Approximate cost to spread the expensive MNIST and spiking jobs across hosts.
    def cost(job):
        samples = job["train_limit"] or (20000 if job["dataset"] == "mnist" else 320)
        model_factor = 20 if job["mode"].startswith("spiking") else 1
        return (
            samples
            * model_factor
            * job["grid"] ** 2
            * job["crop"] ** 2
            * job["kernel"] ** 2
        )

    weights = [3, 1, 1, 1] if count == 4 else [1] * count
    total_weight = sum(weights)
    # Round cumulative quotas so all jobs are assigned and laptop counts differ by at most one.
    boundaries = [0]
    for weight in itertools.accumulate(weights):
        boundaries.append((len(jobs) * weight + total_weight // 2) // total_weight)
    quotas = [end - start for start, end in itertools.pairwise(boundaries)]

    shards = [[] for _ in range(count)]
    costs = [0] * count
    for job in sorted(jobs, key=cost, reverse=True):
        eligible = (i for i in range(count) if len(shards[i]) < quotas[i])
        index = min(eligible, key=lambda i: (costs[i] + cost(job)) / weights[i])
        shards[index].append(job)
        costs[index] += cost(job)
    return shards


def result_path(output, job):
    digest = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
    return (
        output
        / f"{job['dataset']}_{job['mode']}_g{job['grid']}_k{job['kernel']}_s{job['seed']}_{digest}.json"
    )


def write_result(path, result):
    # A crash during writing leaves the final filename absent and the job retryable.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as stream:
        temp = Path(stream.name)
        json.dump(result, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def run_job(job, data, device):
    train, validation = data
    mode = job["mode"]
    seed = job["seed"]
    torch.manual_seed(seed)
    start = time.perf_counter()
    options = dict(epochs=job["epochs"], batch_size=job["batch_size"])
    if mode.startswith("spiking"):
        config = dataset_config(
            job["dataset"],
            seed,
            None,
            5,
            job["kernel"],
            device=device,
            grid=job["grid"],
            reference_frames=mode == "spiking_rf",
        )
        model = Column(config)
        history = model.train(*train, validation_data=validation, **options)
    else:
        crop, centers = fixation_geometry(job["grid"])
        model = PatchCNN(
            job["grid"],
            10 if job["dataset"] == "mnist" else 2,
            position=mode == "cnn_location",
            kernel_size=job["kernel"],
        ).to(device)
        _, history = fit(
            model,
            *train,
            *validation,
            seed=seed,
            prepare=lambda batch: extract_patches(batch, centers, crop),
            **options,
        )
    return dict(
        job=job,
        status="complete",
        accuracy=history[-1]["validation_accuracy"],
        history=history,
        n_train=len(train[1]),
        n_validation=len(validation[1]),
        device=device,
        torch_version=str(torch.__version__),
        seconds=time.perf_counter() - start,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--kernels", nargs="+", type=int, default=[3, 5, 7, 9, 11])
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("caltech", "mnist"),
        default=["caltech", "mnist"],
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--shards",
        type=int,
        default=4,
        help="Four shards allocate half to shard 1 and split the rest across shards 2–4.",
    )
    parser.add_argument("--shard", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cnn-batch-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/grid_kernel_sweep")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        jobs = make_plan(args)
    except ValueError as error:
        parser.error(str(error))
    assigned = partition(jobs, args.shards)[args.shard - 1]
    print(
        f"{len(jobs)} total runs, {len(assigned)} assigned to shard {args.shard}/{args.shards}",
        flush=True,
    )
    pending = []
    for job in assigned:
        path = result_path(args.output, job)
        if path.exists():
            print(f"Completed: {path.name}", flush=True)
        else:
            pending.append((job, path))
            if args.dry_run:
                print(path.name)
    if args.dry_run or not pending:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    for dataset, group in itertools.groupby(
        sorted(pending, key=lambda entry: entry[0]["dataset"]),
        key=lambda entry: entry[0]["dataset"],
    ):
        data = load_data(dataset, args.train_limit, args.val_limit)
        for job, path in group:
            print(f"Starting: {path.name}", flush=True)
            result = run_job(job, data, args.device)
            write_result(path, result)
            print(f"Saved: {path.name} | accuracy={result['accuracy']:.2%}", flush=True)
            gc.collect()


if __name__ == "__main__":
    main()
