"""Build a machine-readable integrity manifest for canonical SCC results."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas
import scipy
import sklearn
import torch
import torchvision
import tqdm

import conex
import pymonntorch

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def aggregate_digest(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def tensor_summary(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().cpu()
    summary = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
    }
    if tensor.numel() and (tensor.is_floating_point() or tensor.is_complex()):
        values = tensor.float()
        summary.update({
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "zero_fraction": float((values == 0).float().mean()),
        })
    return summary


def class_counts(labels: torch.Tensor) -> dict:
    values, counts = torch.unique(labels.long(), return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(values, counts)
    }


def source_records() -> list[dict]:
    paths = []
    for pattern in (
        "scc/**/*.py",
        "scripts/*.py",
        "tests/*.py",
        "notebooks/*.ipynb",
    ):
        paths.extend(ROOT.glob(pattern))
    paths.extend([
        ROOT / "requirements.txt",
        ROOT / "pytest.ini",
        ROOT / "README.md",
        ROOT / "MODEL_AND_EXPERIMENTS_REPRODUCIBILITY.md",
    ])
    return [
        file_record(path)
        for path in sorted(set(paths))
        if path.is_file() and "__pycache__" not in path.parts
    ]


def selected_caltech_records() -> list[dict]:
    roots = [
        ROOT / "data/caltech/101_ObjectCategories",
        ROOT / "data/caltech/caltech101/101_ObjectCategories",
        ROOT / "data/caltech/caltech-101/101_ObjectCategories",
    ]
    category = next(
        (
            path
            for path in roots
            if (path / "Motorbikes").exists()
            and ((path / "Faces_easy").exists() or (path / "Faces").exists())
        ),
        None,
    )
    if category is None:
        return []
    faces = category / "Faces_easy"
    if not faces.exists():
        faces = category / "Faces"
    selected = (
        sorted(faces.glob("*.jpg"))[:200]
        + sorted((category / "Motorbikes").glob("*.jpg"))[:200]
    )
    return [file_record(path) for path in selected]


def artifact_paths() -> list[Path]:
    roots = [
        ROOT / "outputs/joint_training/caltech_all_acquisition_fixation_gated",
        ROOT / "outputs/joint_training/mnist_all_acquisition_fixation_gated",
        ROOT / "outputs/best_models/caltech_fully_spiking_column_no_replay",
        ROOT / "outputs/best_models/mnist_fully_spiking_column_no_replay",
        ROOT / "outputs/l4_kernel_study/caltech",
        ROOT / "outputs/l4_kernel_study/mnist",
        ROOT / "outputs/manuscript_fully_spiking/figures",
        ROOT / "outputs/manuscript_fully_spiking/multiseed_ablation",
        ROOT / "outputs/manuscript_fully_spiking/prospective_reference_frame",
        ROOT / "outputs/manuscript_fully_spiking/caltech_faces_localization_no_replay",
        ROOT / "outputs/manuscript_fully_spiking/caltech_faces_detection",
    ]
    paths = []
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(set(paths))


def fully_spiking_records() -> dict:
    result = {}
    for dataset in ("caltech", "mnist"):
        events_folder = (
            ROOT
            / "outputs/best_models"
            / f"{dataset}_fully_spiking_column_no_replay"
        )
        model_folder = (
            ROOT
            / "outputs/joint_training"
            / f"{dataset}_all_acquisition_fixation_gated"
        )
        events_path = events_folder / "cortical_spike_events.pt"
        model_path = model_folder / "best_fully_spiking_column.pt"
        report_path = model_folder / "fully_spiking_report.json"
        audit_path = model_folder / "fully_spiking_audit.json"
        if not events_path.exists() or not model_path.exists():
            continue
        events = torch.load(events_path, map_location="cpu")
        model = torch.load(model_path, map_location="cpu")
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else {}
        )
        result[dataset] = {
            "events_file": file_record(events_path),
            "model_file": file_record(model_path),
            "report_file": (
                file_record(report_path) if report_path.exists() else None
            ),
            "audit_file": (
                file_record(audit_path) if audit_path.exists() else None
            ),
            "train_events": tensor_summary(events["train_events"]),
            "train_events_binary": events["train_events"].dtype == torch.bool,
            "train_y": {
                **tensor_summary(events["train_y"]),
                "class_counts": class_counts(events["train_y"]),
            },
            "val_events": tensor_summary(events["val_events"]),
            "val_events_binary": events["val_events"].dtype == torch.bool,
            "val_y": {
                **tensor_summary(events["val_y"]),
                "class_counts": class_counts(events["val_y"]),
            },
            "decision_excitatory_weights": tensor_summary(
                model["decision_excitatory_weights"]
            ),
            "decision_inhibitory_weights": tensor_summary(
                model["decision_inhibitory_weights"]
            ),
            "apical_feedback_weights": tensor_summary(
                model["apical_feedback_weights"]
            ),
            "reciprocal_binding_weights": (
                tensor_summary(model["reciprocal_binding_weights"])
                if "reciprocal_binding_weights" in model
                else None
            ),
            "checkpoint_kind": model["kind"],
            "checkpoint_config": model["cfg"],
            "gradient_free": model["gradient_free"],
            "surrogate_derivatives": model["surrogate_derivatives"],
            "decision_learning_rule": model.get("decision_learning_rule"),
            "learning_replays": model.get("learning_replays"),
            "forced_output_spikes": model.get("forced_output_spikes"),
            "raw_image_live_metrics": report.get("raw_image_live_metrics"),
            "event_controls": report.get("event_controls"),
        }
    return result


def environment_record() -> dict:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "memory_bytes": properties.total_memory,
        }
    driver = None
    try:
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception:
        pass
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "conex": getattr(conex, "__version__", "unknown"),
            "pymonntorch": getattr(pymonntorch, "__version__", "unknown"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pandas.__version__,
            "matplotlib": matplotlib.__version__,
            "tqdm": tqdm.__version__,
        },
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "nvidia_driver": driver,
    }


def run(output: Path, verify_against: Path | None = None) -> dict:
    sources = source_records()
    caltech = selected_caltech_records()
    mnist_raw = [
        file_record(path)
        for path in sorted((ROOT / "data/mnist/MNIST/raw").glob("*"))
        if path.is_file()
    ]
    artifacts = [file_record(path) for path in artifact_paths()]
    manifest = {
        "schema_version": 2,
        "root": str(ROOT),
        "environment": environment_record(),
        "source_snapshot": {
            "file_count": len(sources),
            "aggregate_sha256": aggregate_digest(sources),
            "files": sources,
        },
        "datasets": {
            "caltech_selected_400": {
                "file_count": len(caltech),
                "aggregate_sha256": aggregate_digest(caltech),
                "files": caltech,
            },
            "mnist_raw": {
                "file_count": len(mnist_raw),
                "aggregate_sha256": aggregate_digest(mnist_raw),
                "files": mnist_raw,
            },
        },
        "fully_spiking_models": fully_spiking_records(),
        "artifacts": {
            "file_count": len(artifacts),
            "aggregate_sha256": aggregate_digest(artifacts),
            "files": artifacts,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verification = None
    if verify_against is not None:
        expected = json.loads(verify_against.read_text(encoding="utf-8"))
        checks = {
            "source_snapshot": (
                expected["source_snapshot"]["aggregate_sha256"]
                == manifest["source_snapshot"]["aggregate_sha256"]
            ),
            "caltech_selected_400": (
                expected["datasets"]["caltech_selected_400"]["aggregate_sha256"]
                == manifest["datasets"]["caltech_selected_400"]["aggregate_sha256"]
            ),
            "mnist_raw": (
                expected["datasets"]["mnist_raw"]["aggregate_sha256"]
                == manifest["datasets"]["mnist_raw"]["aggregate_sha256"]
            ),
            "artifacts": (
                expected["artifacts"]["aggregate_sha256"]
                == manifest["artifacts"]["aggregate_sha256"]
            ),
        }
        verification = {
            "reference": str(verify_against),
            "checks": checks,
            "all_match": all(checks.values()),
        }
    print(json.dumps({
        "output": str(output),
        "source_sha256": manifest["source_snapshot"]["aggregate_sha256"],
        "caltech_sha256": manifest["datasets"]["caltech_selected_400"][
            "aggregate_sha256"
        ],
        "mnist_sha256": manifest["datasets"]["mnist_raw"]["aggregate_sha256"],
        "artifact_sha256": manifest["artifacts"]["aggregate_sha256"],
        "verification": verification,
    }, indent=2))
    if verification is not None and not verification["all_match"]:
        raise SystemExit(1)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/reproducibility/reproducibility_manifest.json",
    )
    parser.add_argument(
        "--verify-against",
        type=Path,
        default=None,
        help="Compare aggregate hashes with an existing manifest.",
    )
    args = parser.parse_args()
    run(
        args.output.resolve(),
        args.verify_against.resolve() if args.verify_against else None,
    )


if __name__ == "__main__":
    main()
