from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config as cfg
from evaluation_metrics import summarize_numeric


def run_worker(protocol: str, rounds: int, repeat: int, seed: int, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluation_worker.py")),
        "--protocol", protocol,
        "--rounds", str(rounds),
        "--repeat", str(repeat),
        "--seed", str(seed),
        "--output", str(directory),
    ]
    print(f"[Evaluation] Running isolated worker: protocol={protocol}, repeat={repeat + 1}")
    subprocess.run(command, check=True)


def save_plots(rounds: pd.DataFrame, protocol_memory: pd.DataFrame,
               threshold: pd.DataFrame, confidentiality: pd.DataFrame,
               functional: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    clean = rounds

    for metric, ylabel, filename in [
        ("accuracy", "Distributed test accuracy", "accuracy_by_round.png"),
        ("precision", "Precision", "precision_by_round.png"),
        ("recall", "Recall", "recall_by_round.png"),
        ("f1", "F1 score", "f1_by_round.png"),
        ("round_seconds", "Round runtime (seconds)", "runtime_by_round.png"),
        ("round_communication_bytes", "Communication per round (bytes)", "communication_by_round.png"),
    ]:
        plt.figure(figsize=(7.2, 4.4))
        for protocol, group in clean.groupby("protocol"):
            summary = group.groupby("round")[metric].mean()
            plt.plot(summary.index, summary.values, marker="o", label=protocol)
        plt.xlabel("Federated round")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / filename, dpi=200)
        plt.close()

    # Mean phase-time decomposition, excluding common local training in the crypto-only chart.
    phases = ["local_training_seconds", "encryption_seconds", "aggregation_seconds",
              "threshold_decryption_seconds", "query_evaluation_seconds"]
    phase_means = clean.groupby("protocol")[phases].mean()
    ax = phase_means.plot(kind="bar", stacked=True, figsize=(8.0, 4.8))
    ax.set_ylabel("Mean seconds per round")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "runtime_breakdown.png", dpi=200)
    plt.close()

    mem = protocol_memory.set_index("protocol")[["protocol_peak_rss_bytes", "protocol_peak_rss_delta_bytes"]] / (1024**2)
    ax = mem.plot(kind="bar", figsize=(7.0, 4.5))
    ax.set_ylabel("Memory (MiB)")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output / "memory_overhead.png", dpi=200)
    plt.close()

    if not threshold.empty:
        plt.figure(figsize=(6.4, 4.2))
        values = threshold["decryption_succeeded"].astype(int)
        plt.bar(threshold["active_party_count"].astype(str), values)
        plt.axvline(cfg.THRESHOLD - 0.5, linestyle="--", label=f"threshold t={cfg.THRESHOLD}")
        plt.yticks([0, 1], ["failed", "succeeded"])
        plt.xlabel("Active decryption parties")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / "threshold_access_structure.png", dpi=200)
        plt.close()

    if not confidentiality.empty:
        status = confidentiality.groupby("protocol")["passed"].mean()
        plt.figure(figsize=(6.4, 4.2))
        plt.bar(status.index, status.values)
        plt.ylim(0, 1.05)
        plt.ylabel("Passed confidentiality demonstrations")
        plt.tight_layout()
        plt.savefig(output / "confidentiality_demonstration.png", dpi=200)
        plt.close()

    if not functional.empty:
        display = functional.copy()
        display["status"] = display["passed"].astype(int)
        plt.figure(figsize=(8.2, 4.5))
        plt.bar(display["test"], display["status"])
        plt.ylim(0, 1.05)
        plt.ylabel("Test passed")
        plt.xticks(rotation=18, ha="right")
        plt.tight_layout()
        plt.savefig(output / "authorized_query_security.png", dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused evaluation of ordinary FedAvg versus Hybrid PET."
    )
    parser.add_argument("--rounds", type=int, default=cfg.NUM_ROUNDS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("evaluation_results_focused"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output: Path = args.output
    workers = output / "workers"
    plots = output / "plots"
    output.mkdir(parents=True, exist_ok=True)

    for repeat in range(args.repeats):
        for protocol in ("baseline", "hybrid_pet"):
            run_worker(
                protocol, args.rounds, repeat, cfg.RANDOM_SEED,
                workers / f"{protocol}_repeat_{repeat + 1}",
            )

    round_frames: List[pd.DataFrame] = []
    memory_rows: List[dict] = []
    confidentiality_frames: List[pd.DataFrame] = []
    threshold_frames: List[pd.DataFrame] = []
    functional_frames: List[pd.DataFrame] = []

    for directory in sorted(workers.iterdir()):
        round_frames.append(pd.read_csv(directory / "round_metrics.csv"))
        metadata = json.loads((directory / "worker_metadata.json").read_text(encoding="utf-8"))
        memory_rows.append(metadata)
        if (directory / "confidentiality_demo.csv").exists():
            frame = pd.read_csv(directory / "confidentiality_demo.csv")
            frame["repeat"] = metadata["repeat"]
            confidentiality_frames.append(frame)
        if (directory / "threshold_security.csv").exists():
            frame = pd.read_csv(directory / "threshold_security.csv")
            frame["repeat"] = metadata["repeat"]
            threshold_frames.append(frame)
        if (directory / "authorized_query_security.csv").exists():
            frame = pd.read_csv(directory / "authorized_query_security.csv")
            frame["repeat"] = metadata["repeat"]
            functional_frames.append(frame)

    rounds = pd.concat(round_frames, ignore_index=True)
    memory = pd.DataFrame(memory_rows)
    confidentiality = pd.concat(confidentiality_frames, ignore_index=True)
    threshold = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    functional = pd.concat(functional_frames, ignore_index=True) if functional_frames else pd.DataFrame()

    rounds.to_csv(output / "round_metrics.csv", index=False)
    summarize_numeric(rounds, ["protocol", "round"]).to_csv(output / "round_metrics_summary.csv", index=False)
    final = rounds.sort_values("round").groupby(["protocol", "repeat"], as_index=False).tail(1)
    final.to_csv(output / "final_performance_metrics.csv", index=False)
    summarize_numeric(final, ["protocol"]).to_csv(output / "final_performance_summary.csv", index=False)

    runtime_cols = ["protocol", "repeat", "round", "local_training_seconds", "encryption_seconds",
                    "aggregation_seconds", "threshold_decryption_seconds", "query_evaluation_seconds",
                    "round_seconds"]
    rounds[runtime_cols].to_csv(output / "runtime_breakdown.csv", index=False)
    memory.to_csv(output / "memory_overhead.csv", index=False)
    communication_cols = ["protocol", "repeat", "round", "uplink_bytes", "downlink_bytes",
                          "threshold_bytes", "round_communication_bytes"]
    rounds[communication_cols].to_csv(output / "communication_overhead.csv", index=False)
    threshold.to_csv(output / "threshold_security.csv", index=False)
    confidentiality.to_csv(output / "confidentiality_demonstration.csv", index=False)
    functional.to_csv(output / "authorized_query_security.csv", index=False)

    save_plots(rounds, memory, threshold, confidentiality, functional, plots)

    system = {
        "evaluation_scope": [
            "predictive performance", "runtime overhead", "memory overhead",
            "communication overhead", "threshold access structure",
            "server-visible confidentiality demonstration", "authorized query correctness and policy gate",
        ],
        "rounds": args.rounds,
        "repeats": args.repeats,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "methodological_notes": [
            "Baseline and Hybrid PET run in separate fresh subprocesses for fairer peak-RSS comparison.",
            "Both protocols use identical data splits, model initialization, local training, and random seeds.",
            "The confidentiality tests are empirical demonstrations, not formal semantic-security proofs.",
            "The authorization gate models the committee policy that only the registered query is approved; the current core cryptographic code does not itself bind authorization identities to query ciphertexts.",
            "The threshold tests validate the simulated 3-of-5 LISS logic, not physical isolation of shares on separate machines.",
        ],
    }
    (output / "system_metadata.json").write_text(json.dumps(system, indent=2), encoding="utf-8")

    print(f"[Evaluation] Focused results written to: {output.resolve()}")
    for name in [
        "round_metrics.csv", "final_performance_summary.csv", "runtime_breakdown.csv",
        "memory_overhead.csv", "communication_overhead.csv", "threshold_security.csv",
        "confidentiality_demonstration.csv", "authorized_query_security.csv", "system_metadata.json",
    ]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
