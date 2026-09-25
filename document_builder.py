"""Build one isolated set of poisoned documents for an experiment.

This module is deliberately a thin adapter around the existing BiasChain and
baseline agents.  It gives those agents explicit input/output paths and emits a
single, method-independent ``poisoned_docs.csv`` consumed by ``rag_runner.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


METHOD_ALIASES = {
    "auth": "biaschain",
    "biaschain": "biaschain",
    "poisonedrag": "poisonedrag",
    "ebi": "ebi",
    "corruptrag": "corruptrag",
    "simple_prompt_injection": "simple_prompt_injection",
}


def read_table(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
    delimiter = "|" if "|" in header else "," if "," in header else "\t"
    return pd.read_csv(path, sep=delimiter, dtype=str).fillna("")


def normalize_intent_source(source: Path, destination: Path, target_stance: str) -> Path:
    """Normalize an external PoisonedRAG source to intent_agent's pipe CSV."""
    frame = read_table(source)
    if "topic" not in frame.columns:
        if "rephrased_query" in frame.columns:
            frame["topic"] = frame["rephrased_query"]
        elif "original_query" in frame.columns:
            frame["topic"] = frame["original_query"]
        else:
            raise ValueError(f"External source has no topic column: {source}")

    if "idx" not in frame.columns:
        if "query_idx" in frame.columns:
            frame["idx"] = frame["query_idx"]
        else:
            frame["idx"] = range(1, len(frame) + 1)

    if "stance" not in frame.columns:
        frame["stance"] = target_stance
    else:
        matching = frame["stance"].str.upper() == target_stance.upper()
        if not matching.any():
            raise ValueError(
                f"External PoisonedRAG source has no rows for target stance {target_stance}: {source}"
            )
        frame = frame[matching].copy()
    if "statement" not in frame.columns:
        frame["statement"] = ""

    for column in ["intent", "evidence_nodes", "corpus"]:
        if column not in frame.columns:
            frame[column] = ""

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame[
        ["idx", "topic", "statement", "stance", "intent", "evidence_nodes", "corpus"]
    ].drop_duplicates(subset=["idx", "topic"]).to_csv(destination, sep="|", index=False)
    return destination


def compose_poisoned_docs(method: str, output_dir: Path) -> Path:
    method = METHOD_ALIASES[method]

    if method == "biaschain":
        authority = read_table(output_dir / "authority_content.csv")
        coe = read_table(output_dir / "CoE_content.csv")
        merged = authority.merge(coe, on=["idx", "topic", "stance"], how="inner")
        authority_column = "statement"
        if "statement_x" in merged.columns:
            authority_column = "statement_x"
        merged["poisoned_doc"] = (
            merged[authority_column].fillna("") + "\n" + merged["corpus"].fillna("")
        ).str.strip()
        result = merged
    elif method == "poisonedrag":
        result = read_table(output_dir / "PoisonedRAG_results.csv")
        result["poisoned_doc"] = (
            result["topic"].fillna("") + "\n" + result["poisoned"].fillna("")
        ).str.strip()
    elif method == "ebi":
        result = read_table(output_dir / "EBI_results.csv")
        result["poisoned_doc"] = result["poisoned"].fillna("").str.strip()
    elif method in {"corruptrag", "simple_prompt_injection"}:
        filename = (
            "corruptrag_results.csv"
            if method == "corruptrag"
            else "simple_prompt_injection_results.csv"
        )
        result = read_table(output_dir / filename)
        result["poisoned_doc"] = (
            result["topic"].fillna("") + "\n" + result["corpus"].fillna("")
        ).str.strip()
    else:
        raise ValueError(f"Unsupported poisoned document method: {method}")

    if "stance" not in result.columns or "topic" not in result.columns:
        raise ValueError("Poisoned document output must contain topic and stance columns")

    result = result[result["topic"].str.strip().ne("")]
    result = result[result["poisoned_doc"].str.strip().ne("")]
    if result.empty:
        raise ValueError("Document generation produced no usable poisoned documents")

    result = result.drop_duplicates(subset=["topic"], keep="first")
    output_path = output_dir / "poisoned_docs.csv"
    result.to_csv(output_path, sep="|", index=False)

    topics = result["topic"].tolist()
    topic_hash = hashlib.sha256(
        json.dumps(topics, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {
        "method": method,
        "document_count": len(result),
        "topics_sha256": topic_hash,
        "topics": topics,
    }
    (output_dir / "documents_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_path


def build_documents(args: argparse.Namespace) -> Path:
    method = METHOD_ALIASES[args.method]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_stance = args.target_stance.upper()
    intent_path = output_dir / "intent_agent_results.csv"

    if args.source_file and method in {"poisonedrag"}:
        normalize_intent_source(args.source_file.resolve(), intent_path, target_stance)
    elif method not in {"corruptrag", "simple_prompt_injection"} or not args.source_file:
        from intent_agent import run_intent_agent

        run_intent_agent(
            target_stance,
            args.n_questions,
            query_path=args.query_file.resolve(),
            output_dir=output_dir,
            seed=args.seed,
        )

    if method == "biaschain":
        from CoEagent import run_coe_agent
        from authority_agent import run_authority_agent

        run_coe_agent(intent_path, output_dir / "CoE_content.csv")
        run_authority_agent(intent_path, output_dir / "authority_content.csv")
    elif method == "poisonedrag":
        from PoisonedRAG import run_agent

        run_agent(
            target_stance,
            args.poisonedrag_iterations,
            input_path=intent_path,
            output_path=output_dir / "PoisonedRAG_results.csv",
        )

    elif method == "ebi":
        from EBI import run_agent

        run_agent(
            target_stance,
            args.ebi_iterations,
            input_path=intent_path,
            output_path=output_dir / "EBI_results.csv",
            duo_embedding_model=args.ebi_duo_embedding_model,
        )
    elif method == "corruptrag":
        from corruptRAG import run_corruptrag_baseline

        source = args.source_file.resolve() if args.source_file else intent_path
        run_corruptrag_baseline(
            target_stance,
            source_path=str(source),
            output_path=output_dir / "corruptrag_results.csv",
            run_corruptrag_ak=args.run_corruptrag_ak,
        )
    elif method == "simple_prompt_injection":
        from simple_prompt_injection import run_simple_prompt_injection

        source = args.source_file.resolve() if args.source_file else intent_path
        run_simple_prompt_injection(
            target_stance,
            source_path=str(source),
            output_path=output_dir / "simple_prompt_injection_results.csv",
        )
    else:
        raise ValueError(f"Unsupported poisoned document method: {method}")

    return compose_poisoned_docs(method, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_ALIASES), required=True)
    parser.add_argument("--target-stance", choices=["PRO", "CON"], required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-questions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--corpus-file", type=Path, action="append", default=[])
    parser.add_argument("--poisonedrag-iterations", type=int, default=10)
    parser.add_argument("--ebi-iterations", type=int, default=10)
    parser.add_argument("--ebi-fit-corpus-file", type=Path)
    parser.add_argument(
        "--ebi-duo-embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--run-corruptrag-ak",
        action="store_true",
        help="Apply the corruptrag_ak LLM revision step (opt-in via [baseline.corruptrag])",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.query_file.exists() and not args.source_file:
        raise FileNotFoundError(f"Query file does not exist: {args.query_file}")
    if args.source_file and not args.source_file.exists():
        raise FileNotFoundError(f"Baseline source file does not exist: {args.source_file}")
    output = build_documents(args)
    print(f"Saved poisoned documents to: {output}")


if __name__ == "__main__":
    main()
