#!/usr/bin/env python3
"""Multi-GPU, entry-level resumable scheduler for the fixed 10-fold protocol."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--runner", default=None)
    return parser.parse_args()


def entry_complete(entry):
    root = entry["canonical_dir"]
    required = [
        os.path.join(root, "fold_result.json"),
        os.path.join(root, "stage1_predictions.pt"),
        os.path.join(root, "stage2_predictions.pt"),
    ]
    if entry["branch"] == "mil":
        required.extend(
            [
                os.path.join(root, "test_positive_attention.pt"),
                os.path.join(root, "attention_metrics.json"),
            ]
        )
    return all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in required)


def label(entry):
    return f"{entry['dataset_key']}/{entry['branch']}/fold{entry['fold']}"


def main():
    args = parse_args()
    with open(args.manifest, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    root = os.path.dirname(os.path.abspath(args.manifest))
    runner = args.runner or os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_manifest_entry.py")
    gpus = [part.strip() for part in args.gpus.split(",") if part.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")

    entries = [entry for entry in manifest["entries"] if not entry_complete(entry)]
    # Finish cached work first, then spend compute only on genuinely missing Stage-1 fits.
    entries.sort(
        key=lambda entry: (
            entry["stage1"]["status"] != "reuse",
            entry["dataset_key"],
            entry["branch"],
            int(entry["fold"]),
        )
    )
    queue = deque(entries)
    active = {}
    completed = []
    failures = []
    logs = os.path.join(root, "logs")
    os.makedirs(logs, exist_ok=True)
    status_path = os.path.join(root, "orchestrator_status.json")

    def write_status():
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_entries": len(manifest["entries"]),
            "complete_before_start": len(manifest["entries"]) - len(entries),
            "completed_this_run": completed,
            "failed": failures,
            "queued": len(queue),
            "active": {gpu: label(item["entry"]) for gpu, item in active.items()},
            "complete_now": sum(entry_complete(entry) for entry in manifest["entries"]),
        }
        with open(status_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            entry = queue.popleft()
            log_path = os.path.join(
                logs,
                f"{entry['dataset_key']}_{entry['branch']}_fold{entry['fold']}_gpu{gpu}.log",
            )
            log_handle = open(log_path, "a", encoding="utf-8")
            command = [
                sys.executable,
                runner,
                "--manifest",
                args.manifest,
                "--dataset",
                entry["dataset_key"],
                "--branch",
                entry["branch"],
                "--fold",
                str(entry["fold"]),
                "--device",
                "cuda",
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_handle.write(f"\n[{time.strftime('%F %T')}] START {command!r}\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                cwd=os.path.dirname(runner),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[gpu] = {
                "process": process,
                "entry": entry,
                "log": log_handle,
                "log_path": log_path,
            }
        write_status()
        if not active:
            break
        time.sleep(2)
        for gpu, item in list(active.items()):
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["log"].write(f"[{time.strftime('%F %T')}] END rc={returncode}\n")
            item["log"].close()
            entry = item["entry"]
            if returncode == 0 and entry_complete(entry):
                completed.append(label(entry))
            else:
                failures.append(
                    {
                        "entry": label(entry),
                        "returncode": int(returncode),
                        "log": item["log_path"],
                    }
                )
            del active[gpu]

    write_status()
    final = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "complete": sum(entry_complete(entry) for entry in manifest["entries"]),
        "expected": len(manifest["entries"]),
        "failures": failures,
    }
    with open(os.path.join(root, "orchestrator_final.json"), "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2, ensure_ascii=False)
    print(json.dumps(final, ensure_ascii=False))
    raise SystemExit(1 if failures or final["complete"] != final["expected"] else 0)


if __name__ == "__main__":
    main()
