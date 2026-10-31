"""Dataset names and construction settings shared by generation and training."""

import hashlib
from pathlib import Path


_BENCHMARKS = {
    "syn1": (
        "synthetic_milinst_mil_strong_pos_random_v2",
        "478f8cd4092d68e36534b93e3a34d11ae0482b6233746aadd0f0a8fedd645eeb",
    ),
    "syn2": (
        "synthetic_milinst_mil_weak_pos_strong_v2",
        "654256355fef44984fc1cb76099b2e23656565eebe7b833aa42494db5464b38a",
    ),
    "syn3": (
        "synthetic_milinst_both_useful_v2",
        "77365644235e2921cad40367a93a20c58cdaa2f0a07222c6fa7afa14eee91b0f",
    ),
    "igb_tiny__bfs": (
        "igb-tiny_bfs",
        "90dd1a05d1e62a0e5d2853bd76ac991e9a08dd5082e6aa990ea2d39ba4c0e62e",
    ),
    "igb_tiny__metis": (
        "igb-tiny_metis",
        "5e75e1e61ff779792b8597c04bf0025ff80b8677eda9a66202e2285cd97a7592",
    ),
    "igb_tiny__metis_perturb_50": (
        "igb-tiny_metis_perturb_50",
        "e4d5db372c0357a6e639fc5c231165ba72abfd2052786f4483cf2386a18b1731",
    ),
    "ogbn_arxiv__bfs": (
        "ogbn-arxiv_bfs",
        "c191cfb735aebc53733977a444c87ad6066100bfba2a172024d5de58846f95c0",
    ),
    "ogbn_arxiv__metis": (
        "ogbn_arxiv",
        "99cf90ba30485b39e1018560e642f0469fd991316ee1e10ffc3a3af41a523a6c",
    ),
    "ogbn_arxiv__metis_perturb_50": (
        "ogbn_arxiv_metis_perturbed_50",
        "45b2b2125fb903a36d7552eeede49c684b1565649d2b05715281b573132227cf",
    ),
    "ogbn_products__bfs": (
        "ogbn-products_bfs",
        "1cd2d9bf73917b6c4890b1f2a00dd5164f0450cb50226bffa451f32aeffab8c4",
    ),
    "ogbn_products__metis": (
        "ogbn_products",
        "ca69eb2392a973f2528ac66b63bd7b20cf1e3964c8e4684118bb928531a475f8",
    ),
    "ogbn_products__metis_perturb_50": (
        "ogbn_products_metis_perturbed_50",
        "a5ce299291b37f3844e7c403917746ec026c6e21a10a4b8e868d0a72c12de63b",
    ),
    "reddit__bfs": (
        "reddit_bfs",
        "060afaeb2dcbbfa0a0c30bbd2ef5e9e0468024300b9e765acc6743aaa35a78c5",
    ),
    "reddit__metis": (
        "reddit",
        "0c13a786518b19b40cba7cba50b3a6276be92c9df0db58a694f8831d0763dd33",
    ),
    "reddit__metis_perturb_50": (
        "reddit_metis_perturbed_50",
        "4584c35d56c232b4273d8f9574ba072117d407996483583e492c58c6370e0e48",
    ),
}


def benchmark_catalog():
    catalog = {}
    for key, (name, digest) in _BENCHMARKS.items():
        directory = key
        if not key.startswith("syn"):
            dataset, method = key.split("__")
            dataset = "igb-tiny" if dataset == "igb_tiny" else dataset
            method = "metis_perturb" if method == "metis_perturb_50" else method
            directory = str(Path(dataset) / method)
        catalog[key] = {
            "dataset_name": name,
            "directory": directory,
            "processed_sha256": digest,
        }
    return catalog


def real_graph_settings():
    counts = {
        "igb_tiny": 1000,
        "ogbn_arxiv": 2000,
        "ogbn_products": 20000,
        "reddit": 4000,
    }
    settings = {}
    for key in _BENCHMARKS:
        if key.startswith("syn"):
            continue
        dataset, method = key.split("__")
        config = {
            "dataset": dataset.replace("_", "-"),
            "num_partitions": counts[dataset],
            "partition_method": (
                "metis_perturb" if method == "metis_perturb_50" else method
            ),
            "seed": 42,
        }
        if dataset == "igb_tiny":
            config["label_file"] = "node_label_19.npy"
        if method.startswith("metis"):
            config["isolated_ufactor"] = 1.3
        if method == "metis_perturb_50":
            config["perturb_ratio"] = 0.5
        settings[key] = config
    return settings


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
