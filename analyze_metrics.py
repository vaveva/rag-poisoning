"""Bootstrap intervals and paired tests for BiasChain experiment results.

The script consumes completed run directories containing:
  results/clean_annotation.csv
  results/poison_annotation.csv
  results/rag_answers_poison.csv
  state.json

Examples:
  python analyze_metrics.py --runs-root runs_additional_baseline
  python analyze_metrics.py --runs-root runs_additional_baseline --output-dir analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


STANCE_VALUE = {"CON": -1, "NEU": 0, "PRO": 1}
METRICS = ("RSR", "OFSR", "ASV", "TSR", "delta_TSR")
TEST_COLUMNS = [
    "metric",
    "n_topics",
    "biaschain",
    "baseline",
    "difference",
    "randomization_p",
    "randomization_p_holm",
    "wilcoxon_p",
    "wilcoxon_p_holm",
    "biaschain_experiment_id",
    "baseline_experiment_id",
    "dataset",
    "target",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def retrieval_hits(path: Path) -> pd.DataFrame:
    rows = read_csv(path)
    context = rows["retrieved_context"].fillna("").astype(str)
    return rows[["query_idx", "run_idx"]].assign(
        retrieval_hit=context.str.contains(r"Source:\s*tmbcorpus\b", case=False, regex=True).astype(int)
    )


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    results = run_dir / "results"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    experiment = {"experiment_id": state.get("experiment_id", run_dir.name), **state["experiment"]}

    clean = read_csv(results / "clean_annotation.csv").rename(columns={"stance": "clean"})
    poison = read_csv(results / "poison_annotation.csv").rename(columns={"stance": "poison"})
    hits = retrieval_hits(results / "rag_answers_poison.csv")

    keys = ["query_idx", "run_idx"]
    for frame in (clean, poison, hits):
        frame[keys] = frame[keys].astype(int)
    data = clean[keys + ["clean"]].merge(poison[keys + ["poison"]], on=keys, validate="one_to_one")
    data = data.merge(hits, on=keys, validate="one_to_one")
    data["clean"] = data["clean"].str.upper()
    data["poison"] = data["poison"].str.upper()
    data["method"] = experiment["attack_method"]
    data["target"] = str(experiment["target_stance"]).upper()
    return data, experiment


def calculate_metric(data: pd.DataFrame, name: str, target: str) -> float:
    clean = data["clean"]
    poison = data["poison"]
    target = target.upper()

    if name == "RSR":
        return float(data["retrieval_hit"].mean())
    if name == "OFSR":
        eligible = clean != target
        return float((poison[eligible] == target).mean()) if eligible.any() else float("nan")
    if name == "ASV":
        direction = 1 if target == "PRO" else -1
        values = [
            direction * (STANCE_VALUE[p] - STANCE_VALUE[c]) / 2
            for c, p in zip(clean, poison)
        ]
        return float(np.mean(values))
    if name == "TSR":
        return float((poison == target).mean())
    if name == "delta_TSR":
        return float((poison == target).mean() - (clean == target).mean())
    raise ValueError(f"Unknown metric: {name}")


def bootstrap_ci(
    data: pd.DataFrame,
    name: str,
    target: str,
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

    estimates = np.asarray([calculate_metric(sample, name, target) for sample in samples])
    return {
        "estimate": calculate_metric(data, name, target),
        "lower": float(np.nanquantile(estimates, 0.025)),
        "upper": float(np.nanquantile(estimates, 0.975)),
        "bootstrap_unit": unit,
        "bootstrap_repetitions": repetitions,
    }


def topic_metric(data: pd.DataFrame, name: str, target: str) -> pd.DataFrame:
    values = [
        {"query_idx": query_idx, "value": calculate_metric(group, name, target)}
        for query_idx, group in data.groupby("query_idx", sort=True)
    ]
    return pd.DataFrame(values)


def paired_test(
    biaschain: pd.DataFrame,
    baseline: pd.DataFrame,
    name: str,
    target: str,
    repetitions: int,
    rng: np.random.Generator,
) -> dict:
    left = topic_metric(biaschain, name, target).rename(columns={"value": "biaschain"})
    right = topic_metric(baseline, name, target).rename(columns={"value": "baseline"})
    paired = left.merge(right, on="query_idx", validate="one_to_one").dropna()
    observed = paired["biaschain"] - paired["baseline"]

    # A paired randomization test under exchangeability: randomly swap methods
    # within each topic, preserving the dependence between the two methods.
    signs = rng.choice([-1.0, 1.0], size=(repetitions, len(observed)))
    null_means = (signs * observed.to_numpy()).mean(axis=1)
    p_value = float((np.abs(null_means) >= abs(observed.mean())).mean())

    try:
        from scipy.stats import wilcoxon

        wilcoxon_p = float(wilcoxon(observed, zero_method="wilcox").pvalue)
    except (ImportError, ValueError):
        wilcoxon_p = float("nan")

    return {
        "metric": name,
        "n_topics": int(len(observed)),
        "biaschain": float(paired["biaschain"].mean()),
        "baseline": float(paired["baseline"].mean()),
        "difference": float(observed.mean()),
        "randomization_p": p_value,
        "wilcoxon_p": wilcoxon_p,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values while preserving input order."""
    valid = [(index, value) for index, value in enumerate(p_values) if np.isfinite(value)]
    adjusted = [float("nan")] * len(p_values)
    running_max = 0.0
    for rank, (index, value) in enumerate(sorted(valid, key=lambda item: item[1])):
        corrected = min(1.0, (len(valid) - rank) * value)
        running_max = max(running_max, corrected)
        adjusted[index] = running_max
    return adjusted


def discover_runs(runs_root: Path) -> list[Path]:
    return sorted(
        state_path.parent
        for state_path in runs_root.rglob("state.json")
        if state_path.parent.is_dir()
        and (state_path.parent / "results" / "clean_annotation.csv").exists()
        and (state_path.parent / "results" / "poison_annotation.csv").exists()
        and (state_path.parent / "results" / "rag_answers_poison.csv").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_metrics"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    estimates = []
    tests = []
    loaded = []

    for run_dir in discover_runs(args.runs_root):
        try:
            data, experiment = load_run(run_dir)
        except (KeyError, OSError, pd.errors.ParserError, ValueError) as error:
            print(f"Skipping {run_dir}: {error}")
            continue
        loaded.append((run_dir, data, experiment))
        for name in METRICS:
            for unit in ("topic", "query"):
                row = {
                    "experiment_id": run_dir.name,
                    "method": experiment["attack_method"],
                    "target": experiment["target_stance"].upper(),
                    "dataset": experiment["dataset"],
                    "generator_model": experiment["generator_model"],
                    "metric": name,
                    **bootstrap_ci(
                        data,
                        name,
                        experiment["target_stance"],
                        unit,
                        args.bootstrap_repetitions,
                        rng,
                    ),
                }
                estimates.append(row)

    for _, bias_data, bias_experiment in loaded:
        if bias_experiment["attack_method"] != "biaschain":
            continue
        for _, base_data, base_experiment in loaded:
            compatible = all(
                bias_experiment.get(key) == base_experiment.get(key)
                for key in ("dataset", "generator_model", "target_stance", "embedder", "n_questions", "runs_per_question")
            )
            if base_experiment["attack_method"] == "biaschain" or not compatible:
                continue
            for name in METRICS:
                result = paired_test(
                    bias_data,
                    base_data,
                    name,
                    bias_experiment["target_stance"],
                    args.bootstrap_repetitions,
                    rng,
                )
                result.update({
                    "biaschain_experiment_id": bias_experiment["experiment_id"],
                    "baseline": base_experiment["attack_method"],
                    "baseline_experiment_id": base_experiment["experiment_id"],
                    "dataset": bias_experiment["dataset"],
                    "target": bias_experiment["target_stance"],
                })
                tests.append(result)

    methods = sorted({experiment["attack_method"] for _, _, experiment in loaded})
    if "biaschain" not in methods:
        print(
            "Warning: no BiasChain runs were found under "
            f"{args.runs_root}. Found methods: {', '.join(methods) or 'none'}"
        )
    elif not tests:
        print(
            "Warning: BiasChain runs were found, but no compatible baseline "
            "runs matched their dataset/model/target configuration."
        )

    randomization_adjusted = holm_adjust([row["randomization_p"] for row in tests])
    wilcoxon_adjusted = holm_adjust([row["wilcoxon_p"] for row in tests])
    for row, randomization_p, wilcoxon_p in zip(
        tests, randomization_adjusted, wilcoxon_adjusted
    ):
        row["randomization_p_holm"] = randomization_p
        row["wilcoxon_p_holm"] = wilcoxon_p

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(estimates).to_csv(args.output_dir / "bootstrap_intervals.csv", index=False)
    pd.DataFrame(tests, columns=TEST_COLUMNS).to_csv(
        args.output_dir / "biaschain_tests.csv", index=False
    )
    (args.output_dir / "bootstrap_intervals.json").write_text(
        json.dumps(estimates, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Analyzed {len(loaded)} runs")
    print(f"Wrote {args.output_dir / 'bootstrap_intervals.csv'}")
    print(f"Wrote {args.output_dir / 'biaschain_tests.csv'}")


if __name__ == "__main__":
    main()