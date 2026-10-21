"""Calculate retrieval-defense PDR, FPR, and RDR for completed runs.

The script consumes run directories containing:
  state.json
  results/rag_answers_clean.csv
  results/rag_answers_poison.csv

The RAG CSVs must contain the retrieval audit columns written by rag_runner.py.

Examples:
  python analyze_retrieval_metrics.py --runs-root runs_dbscan_filter_corruptrag_ak
  python analyze_retrieval_metrics.py --runs-root runs_dbscan_filter_corruptrag_ak \
      --output-dir analysis_dbscan_retrieval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("PDR", "FPR", "RDR")
KEYS = ["query_idx", "run_idx"]
REQUIRED_COLUMNS = {
    "query_idx",
    "run_idx",
    "poisoned_candidate_count",
    "poisoned_removed_count",
    "clean_candidate_count",
    "clean_removed_count",
    "retrieval_baseline_top_k_count",
    "retrieval_disruption_count",
}


def discover_runs(runs_root: Path) -> list[Path]:
    return sorted(
        state_path.parent
        for state_path in runs_root.rglob("state.json")
        if (state_path.parent / "results" / "rag_answers_clean.csv").exists()
        and (state_path.parent / "results" / "rag_answers_poison.csv").exists()
    )


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    results = run_dir / "results"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    experiment = {
        "experiment_id": state.get("experiment_id", run_dir.name),
        **state["experiment"],
    }

    clean = pd.read_csv(results / "rag_answers_clean.csv")
    poison = pd.read_csv(results / "rag_answers_poison.csv")
    missing = (REQUIRED_COLUMNS - set(clean.columns)) | (
        REQUIRED_COLUMNS - set(poison.columns)
    )
    if missing:
        raise ValueError(
            "missing retrieval audit columns: " + ", ".join(sorted(missing))
        )

    clean = clean[KEYS + ["clean_removed_count", "clean_candidate_count"]]
    poison = poison[
        KEYS
        + [
            "poisoned_removed_count",
            "poisoned_candidate_count",
            "retrieval_disruption_count",
            "retrieval_baseline_top_k_count",
        ]
    ]
    for frame in (clean, poison):
        frame[KEYS] = frame[KEYS].astype(int)

    data = clean.merge(poison, on=KEYS, validate="one_to_one")
    data["query_idx"] = data["query_idx"].astype(int)
    data["method"] = experiment["attack_method"]
    data["target"] = str(experiment["target_stance"]).upper()
    return data, experiment


def calculate_metric(data: pd.DataFrame, name: str) -> float:
    if name == "PDR":
        numerator = data["poisoned_removed_count"].sum()
        denominator = data["poisoned_candidate_count"].sum()
    elif name == "FPR":
        numerator = data["clean_removed_count"].sum()
        denominator = data["clean_candidate_count"].sum()
    elif name == "RDR":
        numerator = data["retrieval_disruption_count"].sum()
        denominator = data["retrieval_baseline_top_k_count"].sum()
    else:
        raise ValueError(f"Unknown retrieval metric: {name}")

    return float(numerator / denominator) if denominator else float("nan")


def bootstrap_ci(
    data: pd.DataFrame,
    name: str,
    unit: str,
    repetitions: int,
    rng: np.random.Generator,
) -> dict:
    if unit == "topic":
        groups = [group for _, group in data.groupby("query_idx", sort=True)]
        samples = [
            pd.concat(
                [groups[index] for index in rng.integers(0, len(groups), len(groups))],
                ignore_index=True,
            )
            for _ in range(repetitions)
        ]
    else:
        indices = rng.integers(0, len(data), size=(repetitions, len(data)))
        samples = [data.iloc[index] for index in indices]

    estimates = np.asarray([calculate_metric(sample, name) for sample in samples])
    return {
        "estimate": calculate_metric(data, name),
        "lower": float(np.nanquantile(estimates, 0.025)),
        "upper": float(np.nanquantile(estimates, 0.975)),
        "bootstrap_unit": unit,
        "bootstrap_repetitions": repetitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis_retrieval_metrics")
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.bootstrap_repetitions < 1:
        parser.error("--bootstrap-repetitions must be positive")

    rng = np.random.default_rng(args.seed)
    loaded = []
    estimates = []
    skipped = 0

    for run_dir in discover_runs(args.runs_root):
        try:
            data, experiment = load_run(run_dir)
        except (KeyError, OSError, ValueError, pd.errors.ParserError) as error:
            skipped += 1
            print(f"Skipping {run_dir}: {error}")
            continue

        loaded.append((data, experiment))
        for name in METRICS:
            for unit in ("topic", "query"):
                estimates.append(
                    {
                        "experiment_id": experiment["experiment_id"],
                        "method": experiment["attack_method"],
                        "target": str(experiment["target_stance"]).upper(),
                        "dataset": experiment["dataset"],
                        "generator_model": experiment["generator_model"],
                        "metric": name,
                        **bootstrap_ci(
                            data,
                            name,
                            unit,
                            args.bootstrap_repetitions,
                            rng,
                        ),
                    }
                )

    summary_rows = []
    grouped = {}
    for data, experiment in loaded:
        group_key = (
            experiment["attack_method"],
            experiment["dataset"],
            str(experiment["target_stance"]).upper(),
        )
        grouped.setdefault(group_key, []).append(data)

    for (method, dataset, target), frames in sorted(grouped.items()):
        data = pd.concat(frames, ignore_index=True)
        for name in METRICS:
            summary_rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "target": target,
                    "metric": name,
                    "estimate": calculate_metric(data, name),
                    "n_runs": len(frames),
                    "n_rows": len(data),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervals_path = args.output_dir / "retrieval_bootstrap_intervals.csv"
    summary_path = args.output_dir / "retrieval_attack_summary.csv"
    json_path = args.output_dir / "retrieval_bootstrap_intervals.json"
    pd.DataFrame(estimates).to_csv(intervals_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(estimates, indent=2) + "\n", encoding="utf-8")

    print(f"Analyzed {len(loaded)} runs; skipped {skipped}")
    print(f"Wrote {intervals_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
