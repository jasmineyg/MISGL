"""NASGL experiments: Internal-Head training followed by Context-Head training."""

import argparse
import hashlib
import json
import logging
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from MISGL.config import (
    Config,
    ContextHeadConfig,
    InternalHeadConfig,
    ModelConfig,
    TrainingConfig,
    default_config,
)
from MISGL.context_head import (
    ContextHead,
    build_context_adjacency,
    normalize_context_adjacency,
    to_sparse_tensor,
)
from MISGL.data import DatasetBundle
from MISGL.model import SubgraphEncoder
from MISGL.trainer import (
    encode_subgraphs,
    evaluate_splits,
    runtime_device,
    set_seed,
    summarize,
    train_context_stage,
    train_internal_stage,
)
from data_construction.dataset_settings import benchmark_catalog


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "nasgl": (True, True),
    "internal": (True, False),
    "gat": (False, False),
    "context": (False, True),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def config_from_dict(raw):
    raw = dict(raw)
    raw["datasets"] = tuple(raw["datasets"])
    for key, cls in (
        ("model", ModelConfig),
        ("training", TrainingConfig),
        ("internal_head", InternalHeadConfig),
        ("context_head", ContextHeadConfig),
    ):
        raw[key] = cls(**raw[key])
    return Config(**raw)


def source_hashes():
    files = list((ROOT / "MISGL").rglob("*.py")) + [ROOT / "train.py"]
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(files)}


def split_fingerprint(bundle, fold_count):
    """Fingerprint generated folds by source sample IDs, independent of files."""
    folds = [
        bundle.sample_indices[bundle.split(fold)["test"]].tolist()
        for fold in range(fold_count)
    ]
    return hashlib.sha256(json.dumps(folds, separators=(",", ":")).encode()).hexdigest()


def context_graph_tensor(bundle, config, device):
    if not config.context_head.enabled:
        return None
    adjacency = build_context_adjacency(
        bundle.original_graph,
        bundle.assignment_matrix,
        bundle.subgraph_ids,
        top_k=config.context_head.top_k,
    )
    return to_sparse_tensor(normalize_context_adjacency(adjacency), device)


def fold_is_complete(folder, fingerprint):
    path = folder / "result.json"
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text())
        required = {"internal_encoder.pt", "predictions.npz"}
        if fingerprint["config"]["context_head"]["enabled"]:
            required.add("context_head.pt")
        return (
            result["status"] == "complete"
            and result["fingerprint"] == fingerprint
            and set(result["artifacts"]) == required
            and all(
                (folder / n).is_file() and sha256(folder / n) == h
                for n, h in result["artifacts"].items()
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_experiment(args):
    entry = benchmark_catalog()[args.benchmark]
    config = default_config()
    use_internal, use_context = VARIANTS[args.variant]
    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else Path(args.data_root) / entry["directory"]
    )
    config = replace(
        config,
        datasets=(entry["dataset_name"],),
        data_dir=str(data_dir.resolve()),
        run_name=args.variant,
        device=args.device or config.device,
        cuda_device=(
            args.cuda_device if args.cuda_device is not None else config.cuda_device
        ),
        internal_head=replace(config.internal_head, enabled=use_internal),
        context_head=replace(config.context_head, enabled=use_context),
    )
    if config.seed != 1024 or config.folds != 10 or config.training.loss != "bce":
        raise ValueError("Paper protocol requires seed=1024, folds=10 and loss=bce")
    if args.smoke:
        config = replace(
            config,
            training=replace(config.training, epochs=1, patience=1),
            context_head=replace(config.context_head, epochs=1, patience=1),
        )
    device = runtime_device(config)
    data_path = data_dir / (entry["dataset_name"] + "_processed.pkl")
    data_hash = sha256(data_path)
    if args.verify_data and data_hash != entry["processed_sha256"]:
        raise ValueError("Processed file differs from the original paper input")
    # Keep the formal preprocessing order and default numerical thread settings.
    bundle = DatasetBundle.load(config, entry["dataset_name"], use_context)
    split_hash = split_fingerprint(bundle, config.folds)
    if args.check_data:
        print(
            json.dumps(
                {
                    "benchmark": args.benchmark,
                    "samples": len(bundle.labels),
                    "feature_dim": bundle.feature_dim,
                    "processed_sha256": data_hash,
                    "matches_original_pickle": data_hash == entry["processed_sha256"],
                    "cv_sha256": split_hash,
                },
                indent=2,
            )
        )
        return None
    output = (
        Path(args.output_dir)
        / ("smoke" if args.smoke else "paper")
        / args.benchmark
        / args.variant
    )
    config = replace(config, output_dir=str(output.resolve()))
    fingerprint = {
        "data": data_hash,
        "split": split_hash,
        "source": source_hashes(),
        "config": json.loads(json.dumps(asdict(config))),
        "smoke": args.smoke,
    }
    metadata = {
        "benchmark": args.benchmark,
        "variant": args.variant,
        "config": asdict(config),
        "fingerprint": fingerprint,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "torch_threads": torch.get_num_threads(),
        },
    }
    manifest = output / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            raise ValueError(
                "Output has different inputs/configuration; choose another --output-dir"
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError(
            "Refusing to overwrite a directory without matching run metadata"
        )
    write_json(manifest, metadata)
    adjacency = context_graph_tensor(bundle, config, device)
    for fold in args.folds:
        folder = output / ("fold_" + str(fold))
        if fold_is_complete(folder, fingerprint):
            logging.info("Skipping verified completed fold %d", fold)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        set_seed(config.seed + fold, device)
        positions = bundle.split(fold)
        loaders = bundle.loaders(fold, config.training.batch_size)
        internal_encoder, internal_selection = train_internal_stage(
            config, bundle.feature_dim, loaders["train"], loaders["val"], device
        )
        internal_logits, z_int, labels, indices = encode_subgraphs(
            internal_encoder, bundle.all_loader(config.training.batch_size), device
        )
        if not torch.equal(indices, torch.as_tensor(bundle.sample_indices)):
            raise RuntimeError("Canonical sample order changed")
        logits, context_head, context_selection = internal_logits, None, None
        if use_context:
            # The original runner continues the Stage-1 RNG stream without reseeding.
            context_head, logits, context_selection = train_context_stage(
                config, z_int, labels, positions, adjacency, device
            )
        torch.save(
            {
                "state_dict": internal_encoder.state_dict(),
                "input_dim": bundle.feature_dim,
                "selection": internal_selection,
            },
            folder / "internal_encoder.pt",
        )
        artifacts = ["internal_encoder.pt", "predictions.npz"]
        if context_head is not None:
            torch.save(
                {
                    "state_dict": context_head.state_dict(),
                    "selection": context_selection,
                },
                folder / "context_head.pt",
            )
            artifacts.append("context_head.pt")
        np.savez_compressed(
            folder / "predictions.npz",
            sample_indices=indices.numpy(),
            labels=labels.numpy(),
            internal_logits=internal_logits.numpy(),
            final_logits=logits.numpy(),
            **positions
        )
        result = {
            "status": "complete",
            "fold": fold,
            "fingerprint": fingerprint,
            "internal_selection": internal_selection,
            "context_selection": context_selection,
            "internal": evaluate_splits(internal_logits, labels, positions),
            "final": evaluate_splits(logits, labels, positions),
            "artifacts": {name: sha256(folder / name) for name in artifacts},
        }
        write_json(folder / "result.json", result)
        logging.info(
            "%s %s fold=%d test_acc=%.4f test_f1=%.4f",
            args.benchmark,
            args.variant,
            fold,
            result["final"]["test"]["acc"],
            result["final"]["test"]["f1"],
        )
    completed = []
    for fold in range(10):
        folder = output / ("fold_" + str(fold))
        if fold_is_complete(folder, fingerprint):
            completed.append(json.loads((folder / "result.json").read_text()))
    summary = {
        "completed_folds": [r["fold"] for r in completed],
        "complete_10fold": len(completed) == 10,
        "smoke": args.smoke,
    }
    if len(completed) == 10 and not args.smoke:
        summary["metrics"] = {
            stage: {
                s: summarize([r[stage][s] for r in completed])
                for s in ("train", "val", "test")
            }
            for stage in ("internal", "final")
        }
    write_json(output / "summary.json", summary)
    return output


def load_run(run_dir, data_dir=None, device_name=None):
    record = json.loads((Path(run_dir) / "run.json").read_text())
    config = config_from_dict(record["config"])
    if data_dir:
        config = replace(config, data_dir=str(Path(data_dir).resolve()))
    if device_name:
        config = replace(config, device=device_name)
    device = runtime_device(config)
    data = Path(config.data_dir) / (config.datasets[0] + "_processed.pkl")
    if sha256(data) != record["fingerprint"]["data"]:
        raise ValueError("Inference data differs from the training run")
    bundle = DatasetBundle.load(config, config.datasets[0], config.context_head.enabled)
    if split_fingerprint(bundle, config.folds) != record["fingerprint"]["split"]:
        raise ValueError(
            "Inference cross-validation folds differ from the training run"
        )
    return record, config, device, bundle


def load_internal_encoder(folder, config, device):
    checkpoint = torch.load(
        Path(folder) / "internal_encoder.pt", map_location="cpu", weights_only=False
    )
    model = SubgraphEncoder(config, checkpoint["input_dim"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def predict_checkpoint(folder, config, device, bundle, adjacency):
    model = load_internal_encoder(folder, config, device)
    internal_logits, z_int, labels, indices = encode_subgraphs(
        model, bundle.all_loader(config.training.batch_size), device
    )
    logits = internal_logits
    if config.context_head.enabled:
        checkpoint = torch.load(
            Path(folder) / "context_head.pt", map_location="cpu", weights_only=False
        )
        context_head = ContextHead(
            config.model.node_dim,
            config.context_head.hidden_dim,
            config.context_head.dropout,
        ).to(device)
        context_head.load_state_dict(checkpoint["state_dict"], strict=True)
        context_head.eval()
        with torch.inference_mode():
            logits = context_head(z_int.to(device), adjacency).detach().cpu()
    return internal_logits, logits, labels, indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="NASGL: final Full-9 paper protocol")
    parser.add_argument(
        "--benchmark", choices=sorted(benchmark_catalog()), default="syn3"
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="nasgl")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-dir", help="Override the processed-pickle directory")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--cuda-device", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=range(10), default=list(range(10))
    )
    parser.add_argument(
        "--verify-data",
        action="store_true",
        help="Require the original processed-pickle SHA-256",
    )
    parser.add_argument(
        "--check-data", action="store_true", help="Validate inputs without training"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="One epoch per stage; stored separately"
    )
    args = parser.parse_args(argv)
    if len(set(args.folds)) != len(args.folds):
        parser.error("--folds must not contain duplicates")
    return args


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    output = run_experiment(parse_args(argv))
    if output is not None:
        print("Run saved to", output)
