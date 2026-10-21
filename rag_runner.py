"""Generic clean/poisoned RAG runner used by the experiment orchestrator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
from typing import Callable, Dict, List, Sequence
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import re


from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
import math


EMBEDDERS = {
    "nomic": "nomic-embed-text",
    "qwen": "qwen3-embedding:4b",
}

AGENT_SYSTEM_PROMPT = (
    "You are a helpful assistant. You have a tool that retrieves context from "
    "an external database. Use it and answer as accurately and objectively as possible."
)

LLM_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the retrieved context to answer the "
    "question as accurately and objectively as possible."
)

@dataclass(frozen=True)
class Phrase:
    doc_index: int
    text: str

class DBSCANDocumentFilter:
    def __init__(
        self,
        embedding_function,
        eps: float,
        min_samples: int,
        metric: str,
        drop_noise: bool,
        min_cluster_size: int,
        keep_if_all_filtered: bool,
    ) -> None:
        if eps <= 0:
            raise ValueError("DBSCAN eps must be positive")
        if min_samples < 1:
            raise ValueError("DBSCAN min_samples must be at least 1")
        if min_cluster_size < 1:
            raise ValueError("DBSCAN min_cluster_size must be at least 1")
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("DBSCAN metric must be 'cosine' or 'euclidean'")

        self.embedding_function = embedding_function
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.drop_noise = drop_noise
        self.min_cluster_size = min_cluster_size
        self.keep_if_all_filtered = keep_if_all_filtered

    def filter(self, documents: Sequence, final_k: int) -> list:
        if len(documents) <= self.min_samples:
            return list(documents[:final_k])

        embeddings = np.asarray(
            self.embedding_function.embed_documents(
                [document.page_content for document in documents]
            ),
            dtype=np.float32,
        )
        if self.metric == "cosine":
            embeddings = normalize(embeddings)

        labels = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
        ).fit_predict(embeddings)
        cluster_sizes = Counter(labels)
        kept = []
        for document, label, retrieval_rank in zip(
            documents,
            labels,
            range(len(documents)),
        ):
            is_noise = label == -1
            is_small_cluster = (
                label != -1 and cluster_sizes[label] < self.min_cluster_size
            )
            if self.drop_noise and is_noise:
                continue
            if is_small_cluster:
                continue
            kept.append((retrieval_rank, document))

        if not kept and self.keep_if_all_filtered:
            kept = list(enumerate(documents))

        kept.sort(key=lambda item: item[0])
        print(
            f"DBSCAN defense removed {len(documents) - len(kept)} "
            f"of {len(documents)} retrieved documents"
        )
        return [document for _, document in kept[:final_k]]


@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    retrieval_rank: int
    retrieval_distance: Optional[float] = None


class DBSCANRetrievalFilter:
    def __init__(
        self,
        eps: float = 0.18,
        min_samples: int = 3,
        metric: str = "cosine",
        drop_noise: bool = True,
        min_cluster_size: int = 2,
        keep_if_all_filtered: bool = True,
    ) -> None:
        if eps <= 0:
            raise ValueError("DBSCAN eps must be positive")
        if min_samples < 1:
            raise ValueError("DBSCAN min_samples must be at least 1")
        if min_cluster_size < 1:
            raise ValueError("DBSCAN min_cluster_size must be at least 1")
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("DBSCAN metric must be 'cosine' or 'euclidean'")

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.drop_noise = drop_noise
        self.min_cluster_size = min_cluster_size
        self.keep_if_all_filtered = keep_if_all_filtered

    def filter(
        self,
        docs: List[RetrievedDoc],
        final_k: int = 5,
    ) -> List[RetrievedDoc]:
        if len(docs) <= self.min_samples:
            return docs[:final_k]

        embeddings = np.vstack([doc.embedding for doc in docs])
        if self.metric == "cosine":
            embeddings = normalize(embeddings)

        labels = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
        ).fit_predict(embeddings)
        cluster_sizes = Counter(labels)
        kept = []
        for doc, label in zip(docs, labels):
            is_noise = label == -1
            is_small_cluster = (
                label != -1 and cluster_sizes[label] < self.min_cluster_size
            )
            if self.drop_noise and is_noise:
                continue
            if is_small_cluster:
                continue
            kept.append(doc)

        if not kept and self.keep_if_all_filtered:
            kept = docs

        kept = sorted(kept, key=lambda doc: doc.retrieval_rank)
        return kept[:final_k]


def calibrate_dbscan_retrieval_filter(
    clean_query_results: List[List[RetrievedDoc]],
    eps_grid,
    min_samples_grid,
    target_false_positive_rate: float = 0.02,
    metric: str = "cosine",
) -> dict:
    """Select the most aggressive DBSCAN setting within a clean FPR budget."""
    if not 0 <= target_false_positive_rate <= 1:
        raise ValueError("target_false_positive_rate must be between 0 and 1")
    if metric not in {"cosine", "euclidean"}:
        raise ValueError("DBSCAN metric must be 'cosine' or 'euclidean'")

    records = []
    for eps in eps_grid:
        for min_samples in min_samples_grid:
            defense = DBSCANRetrievalFilter(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                drop_noise=True,
                min_cluster_size=2,
                keep_if_all_filtered=True,
            )
            total_docs = 0
            removed_docs = 0
            for pool in clean_query_results:
                kept_ids = {doc.doc_id for doc in defense.filter(pool, len(pool))}
                total_docs += len(pool)
                removed_docs += sum(doc.doc_id not in kept_ids for doc in pool)

            records.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "clean_false_positive_rate": (
                        removed_docs / total_docs if total_docs else 1.0
                    ),
                }
            )

    if not records:
        raise ValueError("eps_grid and min_samples_grid must contain values")

    valid = [
        record
        for record in records
        if record["clean_false_positive_rate"] <= target_false_positive_rate
    ]
    if not valid:
        return min(valid or records, key=lambda record: record["clean_false_positive_rate"])
    return min(valid, key=lambda record: record["eps"])

def extract_phrases(text: str, min_words: int) -> List[str]:
    words = normalize_text(text).split()

    if len(words) < min_words:
        return []

    return [
        " ".join(words[start : start + min_words])
        for start in range(len(words) - min_words + 1)
    ]

def _build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        norm="l2",
        dtype=np.float32,
    )

def safe_replace_with_copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Clean and poisoned database paths must differ")
    if source in destination.parents:
        raise ValueError("Poisoned database cannot be placed inside the clean database")
    if destination in {Path("/").resolve(), Path.cwd().resolve(), Path.home().resolve()}:
        raise ValueError(f"Refusing unsafe poisoned database path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def load_jsonl_documents(path: Path, source_prefix: str, document_class):
    documents = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = line
            if isinstance(value, dict):
                text = value.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    text = json.dumps(value, ensure_ascii=False)
                metadata = value.get("metadata", {})
                metadata = metadata.copy() if isinstance(metadata, dict) else {}
            else:
                text = str(value)
                metadata = {}
            metadata.setdefault("source", f"{source_prefix}#L{line_number}")
            documents.append(document_class(page_content=text, metadata=metadata))
    return documents


def load_poisoned_documents(path: Path, document_class):
    documents = []
    topics = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters="|,\t;").delimiter
        except csv.Error:
            delimiter = "|"
        for row in csv.DictReader(handle, delimiter=delimiter):
            topic = (row.get("topic") or "").strip()
            poisoned_text = (row.get("poisoned_doc") or "").strip()
            stance = (row.get("stance") or row.get("target_stance") or "").strip()
            if not topic or not poisoned_text:
                continue
            topics.append(topic)
            documents.append(
                document_class(
                    page_content=poisoned_text,
                    metadata={
                        "source": "tmbcorpus",
                        "poisoned": True,
                        "target_stance": stance,
                    },
                )
            )
    topics = list(dict.fromkeys(topics))
    if not topics:
        raise ValueError(f"No usable topics in {path}")
    return documents, topics


def initialize_generator(provider: str, model_name: str, temperature: float, max_tokens: int, retries: int):
    if provider == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it before starting an unattended run."
            )
        from langchain_openrouter import ChatOpenRouter
        #The parameters temperature and reasoning should be removed if using GPT-5.6 Luna
        return ChatOpenRouter(
            model=model_name,
            #temperature=temperature,
            max_tokens=max_tokens,
            max_retries=retries,
            #reasoning={"effort": "none"},
            openrouter_provider={
                "sort": "throughput",
                "require_parameters": True,
            },
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(model=model_name, temperature=temperature)

def drop_shared_phrase_documents(
    documents: Sequence,
    clusters: Sequence[dict],
    max_documents: int = 30,
) -> list:
    triggered_clusters = []
    blocked_indices = set()

    for cluster in clusters:
        cluster_count = int(cluster.get("count", 0))
        doc_indices = list(cluster.get("doc_indices", []))
        if cluster_count > max_documents:
            triggered_clusters.append(cluster)
            blocked_indices.update(doc_indices)
    kept = [
        document
        for index, document in enumerate(documents)
        if index not in blocked_indices
    ]

    if triggered_clusters:
        print(
            f"Phrase-cluster duplicate defense triggered for "
            f"{len(triggered_clusters)} cluster(s):"
        )
        for cluster in triggered_clusters:
            example = cluster.get("example_phrase", "<unknown>")
            doc_indices = cluster.get("doc_indices", [])
            print(
                f"  - cluster example: {example!r} | "
                f"docs={len(doc_indices)} | "
                f"doc_indices={doc_indices}"
            )
            for doc_index in doc_indices:
                document = documents[doc_index]
                preview = (document.page_content or "").replace("\n", " ")
                preview = " ".join(preview.split())
                print(f"    * dropped doc[{doc_index}]: {preview[:180]}")

    print(
        f"Documents kept: {len(kept)} | "
        f"dropped for widely shared phrases: "
        f"{len(documents) - len(kept)}"
    )
    return kept

def shared_phrase_clusters(
    documents: Sequence,
    similarity_threshold: float = 0.95,
    min_documents: int = 30,
    min_words: int = 10,
) -> List[dict]:
    phrase_documents: dict[str, set[int]] = defaultdict(set)

    for doc_index, document in enumerate(documents):
        # Deduplicate phrases within one document.
        document_phrases = set(
            extract_phrases(document.page_content, min_words)
        )
        for phrase in document_phrases:
            phrase_documents[phrase].add(doc_index)

    shared_phrases = [
        Phrase(doc_index, phrase)
        for phrase, doc_indices in phrase_documents.items()
        if len(doc_indices) >= min_documents
        for doc_index in doc_indices
    ]

    if len(shared_phrases) < 2:
        return []

    vectorizer = _build_vectorizer()
    matrix = vectorizer.fit_transform([phrase.text for phrase in shared_phrases])
    distance_threshold = 1.0 - similarity_threshold

    neighbors = NearestNeighbors(
        metric="cosine",
        radius=distance_threshold,
        n_jobs=-1,
    )
    neighbors.fit(matrix)

    adjacency = neighbors.radius_neighbors_graph(
        matrix,
        mode="connectivity",
    )

    adjacency.setdiag(False)
    adjacency.eliminate_zeros()

    _, labels = connected_components(
        adjacency,
        directed=False,
    )

    clusters_by_label: Dict[int, List[int]] = defaultdict(list)

    for phrase_index, label in enumerate(labels):
        clusters_by_label[int(label)].append(phrase_index)

    clusters = []

    for phrase_indices in clusters_by_label.values():
        doc_indices = sorted(
            {
                shared_phrases[index].doc_index
                for index in phrase_indices
            }
        )

        if len(doc_indices) >= min_documents:
            clusters.append(
                {
                    "example_phrase": shared_phrases[phrase_indices[0]].text,
                    "doc_indices": doc_indices,
                    "count": len(doc_indices),
                }
            )

    clusters.sort(key=lambda cluster: cluster["count"], reverse=True)
    return clusters

def normalize_text(text: str) -> str:
    text = (text or "").strip()
    text = " ".join(text.split())
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

from itertools import combinations

def run(args: argparse.Namespace) -> None:
    import pandas as pd
    from tqdm import tqdm
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not args.poisoned_docs.exists():
        raise FileNotFoundError(f"Poisoned document file does not exist: {args.poisoned_docs}")
    for corpus_file in args.corpus_file:
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file does not exist: {corpus_file}")

    embedding_model = args.embedding_model or EMBEDDERS[args.embedder]
    embedding = OllamaEmbeddings(model=embedding_model)    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        add_start_index=True,
    )

    clean_db = args.clean_db.resolve()
    clean_db.parent.mkdir(parents=True, exist_ok=True)
    clean_store = Chroma(
        collection_name=args.collection_name,
        embedding_function=embedding,
        persist_directory=str(clean_db),
        collection_metadata={"hnsw:space": "cosine"},
    )

    if clean_store._collection.count() == 0:
        raw_documents = []

        for corpus_file in args.corpus_file:
            raw_documents.extend(
                load_jsonl_documents(corpus_file, corpus_file.name, Document)
            )

        if args.duplicate_defense:
            clusters = shared_phrase_clusters(
                raw_documents,
                similarity_threshold=0.80,
                min_documents=10,
                min_words=6,
            )
            raw_documents = drop_shared_phrase_documents(
                raw_documents,
                clusters,
                max_documents=10,
            )

        chunks = splitter.split_documents(raw_documents)
        print(f"Embedding {len(chunks)} chunks")

        for start in tqdm(range(0, len(chunks), args.batch_size),desc="Building clean DB",):
            clean_store.add_documents(chunks[start : start + args.batch_size])
    else:
        print(f"Reusing clean database with {clean_store._collection.count()} documents: {clean_db}")

    poisoned_documents, topics = load_poisoned_documents(args.poisoned_docs, Document)
    if args.duplicate_defense:
        clusters = shared_phrase_clusters(
                poisoned_documents,
                similarity_threshold=0.80,
                min_documents=10,
                min_words=6,
            )
        poisoned_documents = drop_shared_phrase_documents(
                poisoned_documents,
                clusters,
                max_documents=10,
        )

    if args.condition == "poisoned":
        if args.poisoned_db is None:
            raise ValueError("--poisoned-db is required for the poisoned condition")
        safe_replace_with_copy(clean_db, args.poisoned_db)
        active_store = Chroma(
            collection_name=args.collection_name,
            embedding_function=embedding,
            persist_directory=str(args.poisoned_db.resolve()),
        )
        for start in range(0, len(poisoned_documents), args.batch_size):
            active_store.add_documents(poisoned_documents[start : start + args.batch_size])
        print(f"Injected {len(poisoned_documents)} poisoned documents")
    else:
        active_store = clean_store

    if args.dbscan_calibrate:
        calibration_pools = []
        calibration_k = args.top_k * max(5, args.top_k)
        for query_index, query in enumerate(topics):
            clean_results = clean_store.similarity_search_with_score(
                query,
                k=calibration_k,
            )
            clean_documents = [document for document, _ in clean_results]
            clean_embeddings = embedding.embed_documents(
                [document.page_content for document in clean_documents]
            )
            calibration_pools.append(
                [
                    RetrievedDoc(
                        doc_id=f"{query_index}:{rank}",
                        text=document.page_content,
                        embedding=np.asarray(vector, dtype=np.float32),
                        metadata=document.metadata,
                        retrieval_rank=rank,
                        retrieval_distance=float(distance),
                    )
                    for rank, ((document, distance), vector) in enumerate(
                        zip(clean_results, clean_embeddings)
                    )
                ]
            )

        selected_dbscan = calibrate_dbscan_retrieval_filter(
            clean_query_results=calibration_pools,
            eps_grid=args.dbscan_eps_grid,
            min_samples_grid=args.dbscan_min_samples_grid,
            target_false_positive_rate=args.dbscan_target_fpr,
            metric=args.dbscan_metric,
        )
        args.dbscan_eps = float(selected_dbscan["eps"])
        args.dbscan_min_samples = int(selected_dbscan["min_samples"])
        print(f"Calibrated DBSCAN parameters: {selected_dbscan}")

    dbscan_filter = None
    if args.dbscan_defense:
        dbscan_filter = DBSCANDocumentFilter(
            embedding_function=embedding,
            eps=args.dbscan_eps,
            min_samples=args.dbscan_min_samples,
            metric=args.dbscan_metric,
            drop_noise=args.dbscan_drop_noise,
            min_cluster_size=args.dbscan_min_cluster_size,
            keep_if_all_filtered=args.dbscan_keep_if_all_filtered,
        )

    generator = initialize_generator(
        args.generator,
        args.generator_model,
        args.temperature,
        args.max_tokens,
        args.request_retries,
    )

    last_retrieval_metrics = {}

    def calculate_retrieval_metrics(before, after, baseline_top_k):
        after_ids = {id(document) for document in after}
        removed = [document for document in before if id(document) not in after_ids]
        baseline_ids = {id(document) for document in baseline_top_k}
        disrupted = [document for document in baseline_top_k if id(document) not in after_ids]
        poisoned = [document for document in before if document.metadata.get("poisoned", False)]
        clean = [document for document in before if not document.metadata.get("poisoned", False)]
        poisoned_removed = sum(
            id(document) not in after_ids for document in poisoned
        )
        clean_removed = sum(id(document) not in after_ids for document in clean)
        return {
            "retrieval_candidate_count": len(before),
            "retrieval_baseline_top_k_count": len(baseline_top_k),
            "retrieval_kept_count": len(after),
            "retrieval_disruption_count": len(disrupted),
            "retrieval_disrupted": int(bool(disrupted)),
            "poisoned_candidate_count": len(poisoned),
            "poisoned_removed_count": poisoned_removed,
            "clean_candidate_count": len(clean),
            "clean_removed_count": clean_removed,
            "pdr": poisoned_removed / len(poisoned) if poisoned else None,
            "fpr": clean_removed / len(clean) if clean else None,
            "rdr": len(disrupted) / len(baseline_ids) if baseline_ids else 0.0,
        }

    def retrieve(query: str):
        nonlocal last_retrieval_metrics
        print("RETRIEVER QUERY:", repr(query))

        n_buffer = 5
        if args.dbscan_defense:
            fetch_k = args.top_k * max(n_buffer, args.top_k)
        else:
            fetch_k = args.top_k

        retrieved = active_store.similarity_search(query, k=fetch_k)
        baseline_retrieved = list(retrieved)
        baseline_top_k = baseline_retrieved[: args.top_k]

        if dbscan_filter is not None:
            retrieved = dbscan_filter.filter(retrieved, final_k=args.top_k)

        last_retrieval_metrics = calculate_retrieval_metrics(
            baseline_retrieved,
            retrieved,
            baseline_top_k,
        )

        serialized = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
            for doc in retrieved
        )
        return serialized, retrieved

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information from the experiment corpus for the given query."""
        return retrieve(query)

    agent = None
    if args.agentic:
        agent = create_agent(generator, [retrieve_context], system_prompt=AGENT_SYSTEM_PROMPT)

    answers = []
    for query_index, query in enumerate(tqdm(topics, desc="Processing queries"), start=1):
        for run_index in range(1, args.runs_per_question + 1):
            last_retrieval_metrics = {}
            if agent is not None:
                result = agent.invoke({"messages": [{"role": "user", "content": query}]})
                tool_outputs = []
                final_answer = ""
                for message in result["messages"]:
                    message_type = getattr(message, "type", None)
                    content = getattr(message, "content", "")
                    if message_type == "tool":
                        tool_outputs.append(str(content))
                    elif message_type == "ai" and content:
                        final_answer = str(content)
                retrieved_context = "\n\n---\n\n".join(tool_outputs)
            else:
                retrieved_context, _ = retrieve(query)
                messages = [
                    ("system", LLM_SYSTEM_PROMPT),
                    (
                        "human",
                        f"Question:\n{query}\n\nRetrieved context:\n{retrieved_context}",
                    ),
                ]
                result = generator.invoke(messages)
                final_answer = result.content if hasattr(result, "content") else str(result)

            retrieval_metrics = {
                "retrieval_candidate_count": 0,
                "retrieval_baseline_top_k_count": 0,
                "retrieval_kept_count": 0,
                "retrieval_disruption_count": 0,
                "retrieval_disrupted": 0,
                "poisoned_candidate_count": 0,
                "poisoned_removed_count": 0,
                "clean_candidate_count": 0,
                "clean_removed_count": 0,
                "pdr": None,
                "fpr": None,
                "rdr": 0.0,
            }
            retrieval_metrics.update(last_retrieval_metrics)

            answers.append(
                {
                    "original_query": query,
                    "query_idx": query_index,
                    "run_idx": run_index,
                    "rephrased_query": query,
                    "answer": final_answer,
                    "retrieved_context": retrieved_context,
                    **retrieval_metrics,
                }
            )

    expected = len(topics) * args.runs_per_question
    if len(answers) != expected:
        raise RuntimeError(f"Expected {expected} answers, produced {len(answers)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    pd.DataFrame(answers).to_csv(temporary, index=False)
    temporary.replace(args.output)
    print(f"Saved {len(answers)} answers to: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["clean", "poisoned"], required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--corpus-file", type=Path, action="append", required=True)
    parser.add_argument("--poisoned-docs", type=Path, required=True)
    parser.add_argument("--clean-db", type=Path, required=True)
    parser.add_argument("--poisoned-db", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedder", choices=sorted(EMBEDDERS), required=True)
    parser.add_argument("--embedding-model")
    parser.add_argument("--generator", choices=["ollama", "openrouter"], required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--agentic", action="store_true")
    parser.add_argument("--runs-per-question", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--duplicate-defense", action="store_true")
    parser.add_argument("--dbscan-defense", action="store_true")
    parser.add_argument("--dbscan-eps", type=float, default=0.18)
    parser.add_argument("--dbscan-min-samples", type=int, default=3)
    parser.add_argument("--dbscan-metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--dbscan-min-cluster-size", type=int, default=2)
    parser.add_argument("--dbscan-drop-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dbscan-keep-if-all-filtered", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dbscan-calibrate", action="store_true")
    parser.add_argument("--dbscan-eps-grid", type=float, nargs="+", default=[0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.01])
    parser.add_argument("--dbscan-min-samples-grid", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--dbscan-target-fpr", type=float, default=0.02)
    return parser.parse_args()



def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
