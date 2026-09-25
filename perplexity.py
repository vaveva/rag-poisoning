"""End-to-end GPT-2 perplexity analysis for RAG poisoning documents.

The script:
1. Loads one or more labelled ``poisoned_docs.csv`` files.
2. Collects the ordered union of all distinct topic strings.
3. Retrieves one top-ranked clean chunk per topic from a clean Chroma DB and,
   when raw corpus files are supplied, resolves it back to its complete source
   entry before scoring.
4. Calculates GPT-2 perplexity for clean and poisoned documents.
5. Optionally scores either every original clean corpus entry (recommended) or
   every indexed clean Chroma chunk as a corpus reference.
6. Saves per-document scores, group averages, clean pairs, and a multi-panel
   distribution plot comparing each attack method against the clean references.

Example:

    python perplexity_analysis.py \
      --clean-db chroma_db_nomic_natural \
      --collection-name wiki_balance \
      --attack 'BiasChain-PRO=runs/.../documents/poisoned_docs.csv' \
      --attack 'BiasChain-CON=runs/.../documents/poisoned_docs.csv' \
      --attack 'PoisonedRAG-PRO=runs/.../documents/poisoned_docs.csv' \
      --output-dir perplexity/wiki_nomic

To add whole-corpus scores to an already completed analysis without rescoring
the topic-matched and attack documents:

    python perplexity_analysis.py \
      --clean-db chroma_db_nomic_natural \
      --collection-name wiki_balance \
      --output-dir perplexity/wiki_nomic \
      --score-entire-clean-db

Run the script separately for WIKI-BALANCE and ProCon.

For a document-level analysis, provide the same unchunked JSONL corpus file(s)
that were used to build Chroma and enable ``--score-entire-clean-source``. The
retriever still selects the topic-matched comparison, but the selected chunk's
``source`` metadata is used to recover and score the complete JSONL entry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TOPIC_CLEAN_GROUP = "Clean"
WHOLE_CLEAN_GROUP = "Whole Clean Corpus"


@dataclass(frozen=True)
class AttackInput:
    label: str
    path: Path


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters="|,\t;").delimiter
    except csv.Error:
        return "|"


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep=detect_delimiter(path),
        dtype=str,
    ).fillna("")


def parse_attack(value: str) -> AttackInput:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--attack must have the form LABEL=/path/to/poisoned_docs.csv"
        )
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "--attack requires both a non-empty label and path"
        )
    return AttackInput(label=label, path=Path(raw_path).expanduser().resolve())


def load_attack_documents(specification: AttackInput) -> pd.DataFrame:
    frame = read_table(specification.path)
    required = {"topic", "poisoned_doc"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{specification.path} is missing columns: {sorted(missing)}"
        )

    if "idx" not in frame.columns:
        frame["idx"] = range(1, len(frame) + 1)

    frame["topic"] = frame["topic"].str.strip()
    frame["poisoned_doc"] = frame["poisoned_doc"].str.strip()
    frame = frame[
        frame["topic"].ne("") & frame["poisoned_doc"].ne("")
    ].copy()
    frame = frame.drop_duplicates(subset=["topic"], keep="first")
    if frame.empty:
        raise ValueError(f"No usable poisoned documents in {specification.path}")

    frame["group"] = specification.label
    frame["source_file"] = str(specification.path)
    return frame[["idx", "topic", "poisoned_doc", "group", "source_file"]]


def collect_all_topics(attacks: list[pd.DataFrame]) -> list[str]:
    """Return every distinct topic string in first-seen order.

    Small wording variations are intentionally retained. Each variation gets
    its own top-ranked clean document, while every poisoned document remains in
    its requested method distribution.
    """
    topics = []
    seen = set()
    for frame in attacks:
        for topic in frame["topic"]:
            if topic not in seen:
                seen.add(topic)
                topics.append(topic)
    if not topics:
        raise ValueError("The attack files contain no usable topics")
    return topics


def canonical_source_key(value: object) -> str:
    """Normalize ``path/to/file.jsonl#L12`` to ``file.jsonl#L12``."""
    source = str(value or "").strip()
    if "#L" not in source:
        return source
    raw_path, raw_line = source.rsplit("#L", 1)
    try:
        line_number = int(raw_line)
    except ValueError:
        return source
    return f"{Path(raw_path).name}#L{line_number}"


def load_clean_source_entries(paths: list[Path]) -> pd.DataFrame:
    """Load the complete JSONL entries used before retrieval chunking.

    This intentionally mirrors the corpus loader used by ``rag_runner.py``:
    dictionary rows use their ``text`` field, while non-dictionary rows use
    their string representation. Source keys retain the original one-based
    JSONL line number so they match Chroma's ``source`` metadata.
    """
    rows = []
    seen_sources = set()

    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Clean corpus file does not exist: {path}")

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError:
                    value = stripped

                if isinstance(value, dict):
                    text = value.get("text", "")
                    if not isinstance(text, str) or not text.strip():
                        text = json.dumps(value, ensure_ascii=False)
                    metadata = (
                        value.get("metadata", {}).copy()
                        if isinstance(value.get("metadata"), dict)
                        else {}
                    )
                    topic = str(
                        metadata.get("topic")
                        or value.get("topic")
                        or value.get("title")
                        or ""
                    ).strip()
                    document_id = str(value.get("_id", "")).strip()
                    query_id = str(metadata.get("query_id", "")).strip()
                    doc_id = str(metadata.get("doc_id", "")).strip()
                    if not document_id and query_id and doc_id:
                        document_id = f"{query_id}_{doc_id}"
                else:
                    text = str(value)
                    topic = ""
                    document_id = ""
                    query_id = ""
                    doc_id = ""

                text = text.strip()
                if not text:
                    continue

                source_key = f"{path.name}#L{line_number}"
                if source_key in seen_sources:
                    raise ValueError(
                        "Clean corpus filenames must be unique because Chroma "
                        f"stores only the basename in source metadata: {path.name}"
                    )
                seen_sources.add(source_key)
                rows.append(
                    {
                        "idx": len(rows) + 1,
                        "topic": topic,
                        "clean_doc": text,
                        "clean_source": source_key,
                        "source_key": source_key,
                        "source_file": str(path),
                        "document_id": document_id,
                        "query_id": query_id,
                        "doc_id": doc_id,
                    }
                )

    if not rows:
        raise ValueError("The clean corpus files contain no usable text entries")
    return pd.DataFrame(rows)


def open_clean_store(
    clean_db: Path,
    collection_name: str,
    embedding_model: str,
    require_embeddings: bool = True,
):
    try:
        from langchain_chroma import Chroma
    except ImportError as error:
        raise RuntimeError("Clean database access requires langchain-chroma") from error

    if not clean_db.exists():
        raise FileNotFoundError(f"Clean Chroma database does not exist: {clean_db}")

    embedding = None
    if require_embeddings:
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as error:
            raise RuntimeError(
                "Topic-matched clean retrieval requires langchain-ollama"
            ) from error
        embedding = OllamaEmbeddings(model=embedding_model)
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=str(clean_db),
    )
    count = store._collection.count()
    if count <= 0:
        raise RuntimeError(f"Clean Chroma collection is empty: {clean_db}")
    print(f"Clean collection contains {count} documents")
    return store


def build_clean_pairs(
    topics: list[str],
    store,
    source_entries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Retrieve one clean comparison per topic.

    With ``source_entries``, Chroma is used only for selection: the retrieved
    chunk is resolved through its source metadata and GPT-2 receives the full
    unchunked JSONL entry. Without it, legacy chunk-level behavior is retained.
    """

    source_lookup = None
    document_id_lookup = None
    if source_entries is not None:
        source_lookup = source_entries.set_index("source_key", drop=False)
        entries_with_ids = source_entries[
            source_entries["document_id"].astype(str).str.strip().ne("")
        ].copy()
        duplicate_ids = entries_with_ids[
            entries_with_ids["document_id"].duplicated(keep=False)
        ]["document_id"].unique()
        if len(duplicate_ids):
            examples = ", ".join(map(str, duplicate_ids[:3]))
            raise ValueError(
                "Original clean corpus contains duplicate _id values, so "
                f"metadata fallback would be ambiguous. Examples: {examples}"
            )
        document_id_lookup = entries_with_ids.set_index("document_id", drop=False)

    rows = []
    for position, topic in enumerate(topics, start=1):
        retrieved = store.similarity_search(topic, k=1)
        if not retrieved:
            raise RuntimeError(f"No clean document retrieved for topic: {topic}")
        document = retrieved[0]
        clean_source = str(document.metadata.get("source", ""))
        clean_doc = document.page_content
        source_file = clean_source
        source_key = canonical_source_key(clean_source)
        if source_lookup is not None:
            source_row = None
            if source_key and source_key in source_lookup.index:
                source_row = source_lookup.loc[source_key]

            metadata_query_id = str(
                document.metadata.get("query_id", "")
            ).strip()
            metadata_doc_id = str(document.metadata.get("doc_id", "")).strip()
            metadata_document_id = str(
                document.metadata.get("_id")
                or document.metadata.get("document_id")
                or ""
            ).strip()
            if (
                not metadata_document_id
                and metadata_query_id
                and metadata_doc_id
            ):
                metadata_document_id = (
                    f"{metadata_query_id}_{metadata_doc_id}"
                )
            if (
                source_row is None
                and metadata_document_id
                and document_id_lookup is not None
                and metadata_document_id in document_id_lookup.index
            ):
                source_row = document_id_lookup.loc[metadata_document_id]

            if source_row is None:
                raise ValueError(
                    "Could not resolve a retrieved Chroma chunk back to an "
                    "original corpus entry using either its source line or "
                    "query_id/doc_id metadata. "
                    f"source={clean_source!r}, "
                    f"document_id={metadata_document_id!r}. Ensure "
                    "--clean-corpus-file points to the exact JSONL file used "
                    "to build this collection."
                )
            if isinstance(source_row, pd.DataFrame):
                raise ValueError(
                    "Ambiguous original clean entry for retrieved metadata: "
                    f"source={source_key!r}, document_id={metadata_document_id!r}"
                )
            clean_doc = source_row["clean_doc"]
            source_file = source_row["source_file"]
            source_key = source_row["source_key"]
        rows.append(
            {
                "idx": position,
                "topic": topic,
                "clean_doc": clean_doc,
                "clean_source": clean_source,
                "source_key": source_key,
                "source_file": source_file,
            }
        )
        print(f"Retrieved clean pair {position}/{len(topics)}", end="\r", flush=True)
    print()
    return pd.DataFrame(rows)


class GPT2PerplexityScorer:
    def __init__(
        self,
        model_name: str,
        device_name: str,
        stride: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Perplexity scoring requires torch and transformers"
            ) from error

        self.torch = torch
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

        self.device = torch.device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.maximum_length = int(
            getattr(self.model.config, "n_positions", self.tokenizer.model_max_length)
        )
        if stride <= 0 or stride > self.maximum_length:
            raise ValueError(
                f"Stride must be between 1 and {self.maximum_length}, received {stride}"
            )
        self.stride = stride
        print(
            f"Loaded {model_name} on {self.device}; "
            f"maximum context={self.maximum_length}, stride={self.stride}"
        )

    def score(self, text: str) -> tuple[float, float, int]:
        """Return raw PPL, log PPL, and predicted-token count for one document."""
        torch = self.torch
        text = str(text).strip()
        if not text:
            raise ValueError("Cannot calculate perplexity for empty text")

        encoded = self.tokenizer(text, return_tensors="pt")
        all_input_ids = encoded["input_ids"].to(self.device)
        sequence_length = all_input_ids.size(1)
        if sequence_length < 2:
            raise ValueError("Text must contain at least two GPT-2 tokens")

        total_negative_log_likelihood = 0.0
        total_predicted_tokens = 0
        previous_end = 0

        with torch.inference_mode():
            for begin in range(0, sequence_length, self.stride):
                end = min(begin + self.maximum_length, sequence_length)
                target_length = end - previous_end
                input_ids = all_input_ids[:, begin:end]
                target_ids = input_ids.clone()
                target_ids[:, :-target_length] = -100

                output = self.model(input_ids=input_ids, labels=target_ids)
                # Causal-LM loss predicts labels[1:] from input_ids[:-1].
                predicted_tokens = int((target_ids[:, 1:] != -100).sum().item())
                if predicted_tokens > 0:
                    total_negative_log_likelihood += (
                        float(output.loss.item()) * predicted_tokens
                    )
                    total_predicted_tokens += predicted_tokens

                previous_end = end
                if end == sequence_length:
                    break

        if total_predicted_tokens == 0:
            raise ValueError("No tokens were available for causal prediction")

        log_perplexity = total_negative_log_likelihood / total_predicted_tokens
        perplexity = math.exp(log_perplexity)
        return perplexity, log_perplexity, total_predicted_tokens


def score_documents(
    frame: pd.DataFrame,
    text_column: str,
    group: str | None,
    scorer: GPT2PerplexityScorer,
    progress_offset: int,
    progress_total: int,
) -> pd.DataFrame:
    rows = []
    for local_position, (_, row) in enumerate(frame.iterrows(), start=1):
        position = progress_offset + local_position
        label = group if group is not None else str(row["group"])
        ppl, log_ppl, token_count = scorer.score(row[text_column])
        rows.append(
            {
                "idx": row.get("idx", ""),
                "topic": row["topic"],
                "group": label,
                "perplexity": ppl,
                "log_perplexity": log_ppl,
                "token_count": token_count,
                "source_file": row.get("source_file", ""),
            }
        )
        print(
            f"Scored document {position}/{progress_total}: {label}",
            end="\r",
            flush=True,
        )
    return pd.DataFrame(rows)


def score_entire_clean_database(
    store,
    scorer: GPT2PerplexityScorer,
    output_dir: Path,
    clean_db: Path,
    collection_name: str,
    model_name: str,
    stride: int,
    page_size: int,
) -> pd.DataFrame:
    """Score every indexed clean Chroma document with resumable checkpoints."""
    if page_size <= 0:
        raise ValueError("--clean-db-page-size must be greater than zero")

    collection = store._collection
    document_count = int(collection.count())
    score_path = output_dir / "whole_clean_database_scores.csv"
    skipped_path = output_dir / "whole_clean_database_skipped.csv"
    manifest_path = output_dir / "whole_clean_database_manifest.json"
    expected_manifest = {
        "cache_version": 1,
        "clean_db": str(clean_db.resolve()),
        "collection_name": collection_name,
        "document_count": document_count,
        "model_name": model_name,
        "stride": stride,
    }

    if manifest_path.exists():
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatched = [
            key
            for key, value in expected_manifest.items()
            if cached_manifest.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                "Existing whole-clean-database checkpoint does not match the "
                f"current analysis ({', '.join(mismatched)}). Use a different "
                "--output-dir or remove the three whole_clean_database_* cache "
                "files before starting a new corpus analysis."
            )
    elif score_path.exists() or skipped_path.exists():
        raise ValueError(
            "Whole-clean-database cache files exist without their manifest. "
            "Use a different --output-dir or remove those cache files."
        )
    else:
        manifest_path.write_text(
            json.dumps(
                {**expected_manifest, "complete": False, "processed_count": 0},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if score_path.exists():
        existing_scores = pd.read_csv(score_path, dtype={"database_id": str})
    else:
        existing_scores = pd.DataFrame()
    if skipped_path.exists():
        existing_skipped = pd.read_csv(skipped_path, dtype={"database_id": str})
    else:
        existing_skipped = pd.DataFrame()

    processed_ids = set()
    if "database_id" in existing_scores.columns:
        processed_ids.update(existing_scores["database_id"].dropna().astype(str))
    if "database_id" in existing_skipped.columns:
        processed_ids.update(existing_skipped["database_id"].dropna().astype(str))

    if len(processed_ids) > document_count:
        raise ValueError(
            "Whole-clean-database checkpoint contains more IDs than the Chroma "
            "collection; use a new --output-dir."
        )
    if processed_ids:
        print(
            f"Resuming whole clean database scoring: {len(processed_ids)}/"
            f"{document_count} entries already processed"
        )
    if len(processed_ids) == document_count:
        if existing_scores.empty:
            raise RuntimeError(
                "All clean database entries were processed, but none had a "
                "defined GPT-2 perplexity"
            )
        print(
            f"Reusing completed whole clean database scores: "
            f"{len(existing_scores)} documents"
        )
        return existing_scores

    for offset in range(0, document_count, page_size):
        batch = collection.get(
            limit=page_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = [str(value) for value in (batch.get("ids") or [])]
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or [{} for _ in ids]
        if not (len(ids) == len(documents) == len(metadatas)):
            raise RuntimeError(
                "Chroma returned inconsistent IDs, documents, and metadata lengths"
            )

        scored_rows = []
        skipped_rows = []
        for local_position, (database_id, document, metadata) in enumerate(
            zip(ids, documents, metadatas), start=1
        ):
            if database_id in processed_ids:
                continue
            metadata = metadata or {}
            try:
                ppl, log_ppl, token_count = scorer.score(document or "")
            except ValueError as error:
                skipped_rows.append(
                    {
                        "database_id": database_id,
                        "database_position": offset + local_position,
                        "reason": str(error),
                    }
                )
            else:
                scored_rows.append(
                    {
                        "idx": offset + local_position,
                        "topic": "",
                        "group": WHOLE_CLEAN_GROUP,
                        "perplexity": ppl,
                        "log_perplexity": log_ppl,
                        "token_count": token_count,
                        "source_file": metadata.get("source", ""),
                        "database_id": database_id,
                    }
                )

        if scored_rows:
            pd.DataFrame(scored_rows).to_csv(
                score_path,
                mode="a",
                header=not score_path.exists(),
                index=False,
            )
        if skipped_rows:
            pd.DataFrame(skipped_rows).to_csv(
                skipped_path,
                mode="a",
                header=not skipped_path.exists(),
                index=False,
            )

        processed_ids.update(row["database_id"] for row in scored_rows)
        processed_ids.update(row["database_id"] for row in skipped_rows)
        manifest_path.write_text(
            json.dumps(
                {
                    **expected_manifest,
                    "complete": len(processed_ids) == document_count,
                    "processed_count": len(processed_ids),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Scored whole clean database {len(processed_ids)}/{document_count}",
            end="\r",
            flush=True,
        )

    print()
    if len(processed_ids) != document_count:
        raise RuntimeError(
            f"Only {len(processed_ids)} of {document_count} Chroma entries were "
            "processed"
        )
    if not score_path.exists():
        raise RuntimeError("No clean database documents were valid for GPT-2 scoring")

    scores = pd.read_csv(score_path, dtype={"database_id": str})
    skipped_count = len(existing_skipped)
    if skipped_path.exists():
        skipped_count = len(pd.read_csv(skipped_path))
    print(
        f"Whole clean database complete: {len(scores)} scored, "
        f"{skipped_count} skipped because perplexity was undefined"
    )
    return scores


def score_entire_clean_sources(
    source_entries: pd.DataFrame,
    scorer: GPT2PerplexityScorer,
    output_dir: Path,
    corpus_files: list[Path],
    model_name: str,
    stride: int,
    checkpoint_size: int,
) -> pd.DataFrame:
    """Score complete pre-chunking corpus entries with resumable checkpoints."""
    if checkpoint_size <= 0:
        raise ValueError("--clean-source-checkpoint-size must be greater than zero")

    score_path = output_dir / "whole_clean_source_scores.csv"
    skipped_path = output_dir / "whole_clean_source_skipped.csv"
    manifest_path = output_dir / "whole_clean_source_manifest.json"
    resolved_files = [path.expanduser().resolve() for path in corpus_files]
    file_descriptors = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "modified_time_ns": path.stat().st_mtime_ns,
        }
        for path in resolved_files
    ]
    expected_manifest = {
        "cache_version": 1,
        "corpus_files": file_descriptors,
        "document_count": len(source_entries),
        "model_name": model_name,
        "stride": stride,
    }

    if manifest_path.exists():
        cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatched = [
            key
            for key, value in expected_manifest.items()
            if cached_manifest.get(key) != value
        ]
        if mismatched:
            raise ValueError(
                "Existing whole-clean-source checkpoint does not match the "
                f"current analysis ({', '.join(mismatched)}). Use a different "
                "--output-dir or remove the three whole_clean_source_* cache "
                "files before starting a new source-document analysis."
            )
    elif score_path.exists() or skipped_path.exists():
        raise ValueError(
            "Whole-clean-source cache files exist without their manifest. "
            "Use a different --output-dir or remove those cache files."
        )
    else:
        manifest_path.write_text(
            json.dumps(
                {**expected_manifest, "complete": False, "processed_count": 0},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    existing_scores = (
        pd.read_csv(score_path, dtype={"source_key": str})
        if score_path.exists()
        else pd.DataFrame()
    )
    existing_skipped = (
        pd.read_csv(skipped_path, dtype={"source_key": str})
        if skipped_path.exists()
        else pd.DataFrame()
    )
    processed_sources = set()
    if "source_key" in existing_scores.columns:
        processed_sources.update(
            existing_scores["source_key"].dropna().astype(str)
        )
    if "source_key" in existing_skipped.columns:
        processed_sources.update(
            existing_skipped["source_key"].dropna().astype(str)
        )

    expected_sources = set(source_entries["source_key"].astype(str))
    unknown_sources = processed_sources - expected_sources
    if unknown_sources:
        raise ValueError(
            "Whole-clean-source checkpoint contains entries not present in "
            "the supplied corpus files; use a new --output-dir."
        )
    if processed_sources:
        print(
            f"Resuming whole clean source scoring: {len(processed_sources)}/"
            f"{len(source_entries)} entries already processed"
        )
    if processed_sources == expected_sources:
        if existing_scores.empty:
            raise RuntimeError(
                "All clean source entries were processed, but none had a "
                "defined GPT-2 perplexity"
            )
        print(
            f"Reusing completed whole clean source scores: "
            f"{len(existing_scores)} documents"
        )
        return existing_scores

    scored_rows = []
    skipped_rows = []

    def checkpoint() -> None:
        nonlocal scored_rows, skipped_rows
        if scored_rows:
            pd.DataFrame(scored_rows).to_csv(
                score_path,
                mode="a",
                header=not score_path.exists(),
                index=False,
            )
        if skipped_rows:
            pd.DataFrame(skipped_rows).to_csv(
                skipped_path,
                mode="a",
                header=not skipped_path.exists(),
                index=False,
            )
        processed_sources.update(row["source_key"] for row in scored_rows)
        processed_sources.update(row["source_key"] for row in skipped_rows)
        manifest_path.write_text(
            json.dumps(
                {
                    **expected_manifest,
                    "complete": processed_sources == expected_sources,
                    "processed_count": len(processed_sources),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        scored_rows = []
        skipped_rows = []

    for _, row in source_entries.iterrows():
        source_key = str(row["source_key"])
        if source_key in processed_sources:
            continue
        try:
            ppl, log_ppl, token_count = scorer.score(row["clean_doc"])
        except ValueError as error:
            skipped_rows.append(
                {
                    "source_key": source_key,
                    "reason": str(error),
                }
            )
        else:
            scored_rows.append(
                {
                    "idx": row["idx"],
                    "topic": row["topic"],
                    "group": WHOLE_CLEAN_GROUP,
                    "perplexity": ppl,
                    "log_perplexity": log_ppl,
                    "token_count": token_count,
                    "source_file": row["source_file"],
                    "source_key": source_key,
                }
            )

        buffered = len(scored_rows) + len(skipped_rows)
        if buffered >= checkpoint_size:
            checkpoint()
            print(
                f"Scored whole clean sources {len(processed_sources)}/"
                f"{len(source_entries)}",
                end="\r",
                flush=True,
            )

    checkpoint()
    print()
    if processed_sources != expected_sources:
        raise RuntimeError(
            f"Only {len(processed_sources)} of {len(source_entries)} original "
            "clean entries were processed"
        )
    if not score_path.exists():
        raise RuntimeError("No clean source documents were valid for GPT-2 scoring")

    scores = pd.read_csv(score_path, dtype={"source_key": str})
    skipped_count = 0
    if skipped_path.exists():
        skipped_count = len(pd.read_csv(skipped_path))
    print(
        f"Whole clean source corpus complete: {len(scores)} scored, "
        f"{skipped_count} skipped because perplexity was undefined"
    )
    return scores


def make_summary(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("group", as_index=False)
        .agg(
            average_perplexity=("perplexity", "mean"),
            median_perplexity=("perplexity", "median"),
            standard_deviation_perplexity=("perplexity", "std"),
            average_log_perplexity=("log_perplexity", "mean"),
            median_log_perplexity=("log_perplexity", "median"),
            document_count=("perplexity", "size"),
            average_token_count=("token_count", "mean"),
        )
        .sort_values("group")
        .reset_index(drop=True)
    )


def save_analysis_outputs(
    scores: pd.DataFrame,
    output_dir: Path,
    full_plot_range: bool = False,
) -> None:
    scores.to_csv(output_dir / "perplexity_scores.csv", index=False)
    summary = make_summary(scores)
    summary.to_csv(output_dir / "perplexity_summary.csv", index=False)
    print("\nPerplexity summary:")
    print(summary.to_string(index=False))
    plot_distributions(
        scores,
        output_dir / "perplexity_distribution.pdf",
        full_range=full_plot_range,
    )


def plot_distributions(
    scores: pd.DataFrame,
    output_path: Path,
    full_range: bool = False,
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as error:
        raise RuntimeError(
            "Distribution plotting requires matplotlib and seaborn"
        ) from error

    clean = scores[scores["group"] == TOPIC_CLEAN_GROUP]
    whole_clean = scores[scores["group"] == WHOLE_CLEAN_GROUP]
    attack_groups = [
        group
        for group in pd.unique(scores["group"])
        if group not in {TOPIC_CLEAN_GROUP, WHOLE_CLEAN_GROUP}
    ]
    if clean.empty:
        raise ValueError("Distribution plot requires a Clean group")
    if not attack_groups:
        raise ValueError("Distribution plot requires at least one attack group")

    # With the expected four methods this creates the same 1 x 4 structure used
    # in FlippedRAG and Topic-FlipRAG. It also remains usable with fewer methods.
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(
        1,
        len(attack_groups),
        figsize=(5.0 * len(attack_groups), 4.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    whole_clean_color = "#4C72B0"
    topic_clean_color = "#DD8452" if not whole_clean.empty else "#4C72B0"
    attack_color = "#55A868"

    for position, (axis, group) in enumerate(zip(axes.flat, attack_groups)):
        attack = scores[scores["group"] == group]
        if not whole_clean.empty:
            sns.kdeplot(
                data=whole_clean,
                x="log_perplexity",
                color=whole_clean_color,
                fill=True,
                alpha=0.16,
                linewidth=1.6,
                label="Whole clean corpus",
                warn_singular=False,
                ax=axis,
            )
        sns.kdeplot(
            data=clean,
            x="log_perplexity",
            color=topic_clean_color,
            fill=True,
            alpha=0.20,
            linewidth=1.8,
            label="Topic-matched clean" if not whole_clean.empty else "Clean",
            warn_singular=False,
            ax=axis,
        )
        sns.kdeplot(
            data=attack,
            x="log_perplexity",
            color=attack_color,
            fill=True,
            alpha=0.22,
            linewidth=1.8,
            label=group,
            warn_singular=False,
            ax=axis,
        )
        axis.set_title(group)
        axis.set_xlabel("Log PPL")
        axis.set_ylabel("")
        axis.legend(loc="upper right", frameon=True, fontsize=8)

    if not full_range:
        plot_values = scores[["group", "log_perplexity"]].copy()
        plot_values["log_perplexity"] = pd.to_numeric(
            plot_values["log_perplexity"], errors="coerce"
        )
        plot_values = plot_values.dropna(subset=["log_perplexity"])
        plot_values = plot_values[
            plot_values["log_perplexity"].map(math.isfinite)
        ]
        if len(plot_values) >= 2:
            group_limits = plot_values.groupby("group")["log_perplexity"].quantile(
                [0.001, 0.999]
            ).unstack()
            lower = float(group_limits[0.001].min())
            upper = float(group_limits[0.999].max())
            if upper > lower:
                padding = 0.04 * (upper - lower)
                axes.flat[0].set_xlim(lower - padding, upper + padding)
                print(
                    "Distribution x-axis covers the central 99.8% within each "
                    f"group ({lower:.3f} to {upper:.3f}); all values remain "
                    "included in scoring and KDE estimation"
                )

    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attack",
        action="append",
        type=parse_attack,
        help=(
            "Labelled poisoned-doc file as LABEL=PATH; repeat for every attack "
            "and stance. Reusing a label combines files into one distribution."
        ),
    )
    parser.add_argument("--clean-db", type=Path)
    parser.add_argument("--collection-name")
    parser.add_argument(
        "--clean-corpus-file",
        action="append",
        type=Path,
        help=(
            "Original unchunked JSONL corpus used to build Chroma; repeat if "
            "the clean collection was built from multiple files. When given, "
            "topic-matched clean chunks are resolved back to these complete "
            "source entries before GPT-2 scoring."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Regenerate the distribution PDF and PNG from an existing "
            "OUTPUT_DIR/perplexity_scores.csv without rescoring documents"
        ),
    )
    parser.add_argument(
        "--full-plot-range",
        action="store_true",
        help=(
            "Show the complete log-PPL x-axis instead of zooming to the "
            "central 99.8 percent of values"
        ),
    )
    parser.add_argument(
        "--score-entire-clean-db",
        action="store_true",
        help=(
            "Score every indexed clean Chroma document and add a Whole Clean "
            "Corpus group. Without --attack, extend the existing scores in "
            "OUTPUT_DIR instead of rescoring topic-matched and attack documents."
        ),
    )
    parser.add_argument(
        "--score-entire-clean-source",
        action="store_true",
        help=(
            "Score every complete entry in --clean-corpus-file and add it as "
            "the Whole Clean Corpus group. This is the recommended mode for "
            "document-level clean-versus-poisoned comparison."
        ),
    )
    parser.add_argument(
        "--clean-db-page-size",
        type=int,
        default=500,
        help=(
            "Number of Chroma documents fetched and checkpointed at a time "
            "when --score-entire-clean-db is enabled"
        ),
    )
    parser.add_argument(
        "--clean-source-checkpoint-size",
        type=int,
        default=50,
        help=(
            "Number of original clean source entries scored between resumable "
            "checkpoint writes"
        ),
    )
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding-window stride for documents longer than GPT-2's context",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.score_entire_clean_db and args.score_entire_clean_source:
        raise ValueError(
            "Choose only one whole-clean reference: "
            "--score-entire-clean-db or --score-entire-clean-source"
        )

    if args.plot_only:
        if args.score_entire_clean_db or args.score_entire_clean_source:
            raise ValueError(
                "--plot-only cannot be combined with a whole-clean scoring mode"
            )
        scores_path = output_dir / "perplexity_scores.csv"
        scores = pd.read_csv(scores_path)
        required = {"group", "log_perplexity"}
        missing = required - set(scores.columns)
        if missing:
            raise ValueError(
                f"{scores_path} is missing columns required for plotting: "
                f"{sorted(missing)}"
            )
        plot_distributions(
            scores,
            output_dir / "perplexity_distribution.pdf",
            full_range=args.full_plot_range,
        )
        print(f"Regenerated distribution plot in: {output_dir}")
        return

    if args.score_entire_clean_db and not args.attack:
        missing_arguments = []
        if args.clean_db is None:
            missing_arguments.append("--clean-db")
        if not args.collection_name:
            missing_arguments.append("--collection-name")
        if missing_arguments:
            raise ValueError(
                "Extending an existing analysis requires: "
                + ", ".join(missing_arguments)
            )

        scores_path = output_dir / "perplexity_scores.csv"
        if not scores_path.exists():
            raise FileNotFoundError(
                "No existing perplexity_scores.csv was found in "
                f"{output_dir}; provide --attack inputs to run the full analysis"
            )
        existing_scores = pd.read_csv(scores_path)
        existing_scores = existing_scores[
            existing_scores["group"] != WHOLE_CLEAN_GROUP
        ].copy()
        clean_db = args.clean_db.expanduser().resolve()
        store = open_clean_store(
            clean_db=clean_db,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            require_embeddings=False,
        )
        scorer = GPT2PerplexityScorer(
            model_name=args.model_name,
            device_name=args.device,
            stride=args.stride,
        )
        whole_clean_scores = score_entire_clean_database(
            store=store,
            scorer=scorer,
            output_dir=output_dir,
            clean_db=clean_db,
            collection_name=args.collection_name,
            model_name=args.model_name,
            stride=args.stride,
            page_size=args.clean_db_page_size,
        )
        scores = pd.concat(
            [existing_scores, whole_clean_scores], ignore_index=True, sort=False
        )
        save_analysis_outputs(
            scores,
            output_dir,
            full_plot_range=args.full_plot_range,
        )
        print(f"\nAdded whole clean corpus analysis to: {output_dir}")
        return

    missing_arguments = []
    if not args.attack:
        missing_arguments.append("--attack")
    if args.clean_db is None:
        missing_arguments.append("--clean-db")
    if not args.collection_name:
        missing_arguments.append("--collection-name")
    if args.score_entire_clean_source and not args.clean_corpus_file:
        missing_arguments.append("--clean-corpus-file")
    if missing_arguments:
        raise ValueError(
            "Full analysis requires: " + ", ".join(missing_arguments)
        )

    attack_frames = [load_attack_documents(item) for item in args.attack]
    topics = collect_all_topics(attack_frames)
    stale_exclusions = output_dir / "excluded_topics.csv"
    if stale_exclusions.exists():
        stale_exclusions.unlink()
    print(
        f"Loaded {len(attack_frames)} attack files with "
        f"{len(topics)} distinct topic strings; all poisoned documents retained"
    )

    clean_db = args.clean_db.expanduser().resolve()
    source_entries = None
    if args.clean_corpus_file:
        source_entries = load_clean_source_entries(args.clean_corpus_file)
        print(
            f"Loaded {len(source_entries)} complete clean source entries from "
            f"{len(args.clean_corpus_file)} corpus file(s)"
        )
    store = open_clean_store(
        clean_db=args.clean_db.expanduser().resolve(),
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
    )
    clean_pairs = build_clean_pairs(
        topics=topics,
        store=store,
        source_entries=source_entries,
    )
    clean_pairs.to_csv(output_dir / "clean_topic_pairs.csv", sep="|", index=False)

    scorer = GPT2PerplexityScorer(
        model_name=args.model_name,
        device_name=args.device,
        stride=args.stride,
    )

    combined_attacks = pd.concat(attack_frames, ignore_index=True)
    total_documents = len(clean_pairs) + len(combined_attacks)
    clean_scores = score_documents(
        clean_pairs,
        text_column="clean_doc",
        group=TOPIC_CLEAN_GROUP,
        scorer=scorer,
        progress_offset=0,
        progress_total=total_documents,
    )
    attack_scores = score_documents(
        combined_attacks,
        text_column="poisoned_doc",
        group=None,
        scorer=scorer,
        progress_offset=len(clean_pairs),
        progress_total=total_documents,
    )
    print()

    score_frames = [clean_scores, attack_scores]
    if args.score_entire_clean_db:
        score_frames.append(
            score_entire_clean_database(
                store=store,
                scorer=scorer,
                output_dir=output_dir,
                clean_db=clean_db,
                collection_name=args.collection_name,
                model_name=args.model_name,
                stride=args.stride,
                page_size=args.clean_db_page_size,
            )
        )
    if args.score_entire_clean_source:
        if source_entries is None:
            raise RuntimeError(
                "--score-entire-clean-source requires --clean-corpus-file"
            )
        score_frames.append(
            score_entire_clean_sources(
                source_entries=source_entries,
                scorer=scorer,
                output_dir=output_dir,
                corpus_files=args.clean_corpus_file,
                model_name=args.model_name,
                stride=args.stride,
                checkpoint_size=args.clean_source_checkpoint_size,
            )
        )
    scores = pd.concat(score_frames, ignore_index=True, sort=False)
    save_analysis_outputs(
        scores,
        output_dir,
        full_plot_range=args.full_plot_range,
    )
    print(f"\nSaved analysis to: {output_dir}")


if __name__ == "__main__":
    main()