"""Resumable experiment orchestrator for the BiasChain evaluation pipeline.

The orchestrator is intentionally standard-library only.  Heavy pipeline
dependencies are loaded in isolated child processes so one failed experiment
does not corrupt or stop the rest of the matrix.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGES = ["documents", "poisoned_rag", "clean_rag", "evaluation"]
METHOD_ALIASES = {
    "auth": "biaschain",
    "biaschain": "biaschain",
    "poisonedrag": "poisonedrag",
    "ebi": "ebi",
    "corruptrag": "corruptrag",
    "simple_prompt_injection": "simple_prompt_injection",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value: Any, length: int = 16) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def path_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    if path.is_file() and stat.st_size <= 20 * 1024 * 1024:
        return {"path": str(path), "size": stat.st_size, "sha256": file_sha256(path)}
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class Experiment:
    dataset: str
    generator: str
    generator_model: str
    attack_method: str
    target_stance: str
    embedder: str
    embedding_model: str
    agentic_rag: bool
    n_questions: int
    runs_per_question: int
    seed: int
    top_k: int
    input_signature: str
    stance_model: str
    code_signature: str

    @property
    def slug(self) -> str:
        agent = "agent" if self.agentic_rag else "base"
        parts = [self.dataset, self.generator, self.attack_method, self.target_stance.lower(), self.embedder, agent]
        return "-".join(part.replace("_", "-") for part in parts)

    @property
    def experiment_id(self) -> str:
        return f"{self.slug}-{digest(asdict(self), 10)}"


class ConfigError(ValueError):
    pass


class StageFailure(RuntimeError):
    pass


class Orchestrator:
    def __init__(self, config_path: Path, no_cache: bool = False):
        self.config_path = config_path.resolve()
        self.root = self.config_path.parent
        with self.config_path.open("rb") as handle:
            self.config = tomllib.load(handle)
        self.execution = self.config.get("execution", {})
        self.datasets = self.config.get("dataset", {})
        self.models = self.config.get("model", {})
        self.embedders = self.config.get("embedder", {})
        self.baselines = self.config.get("baseline", {})
        self.matrix = self.config.get("matrix", {})
        self.runs_dir = resolve_path(self.root, self.execution.get("runs_dir", "runs"))
        self.cache_dir = resolve_path(self.root, self.execution.get("cache_dir", ".biaschain_cache"))
        self.python = self._python_command(self.execution.get("python", sys.executable))
        self.max_retries = int(self.execution.get("max_retries", 2))
        self.retry_backoff = float(self.execution.get("retry_backoff_seconds", 30))
        timeout_minutes = float(self.execution.get("stage_timeout_minutes", 0))
        self.timeout = timeout_minutes * 60 if timeout_minutes > 0 else None
        self.fail_fast = bool(self.execution.get("fail_fast", False))
        self.cache_enabled = not no_cache
        self.script_dir = Path(__file__).resolve().parent
        self._run_dir_cache: dict[str, Path] = {}
        tracked_code = [
            "document_builder.py",
            "rag_runner.py",
            "stats.py",
            "intent_agent.py",
            "CoEagent.py",
            "authority_agent.py",
            "PoisonedRAG.py",
            "EBI.py",
            "duo.py",
            "corruptRAG.py",
            "simple_prompt_injection.py",
        ]
        self.code_signature = digest(
            [path_signature(self.script_dir / name) for name in tracked_code], 16
        )
        self._validate()

    def _python_command(self, value: str) -> str:
        if value == "python" or value == "python3" or Path(value).is_absolute():
            return value
        candidate = resolve_path(self.root, value)
        return str(candidate)

    def _validate(self) -> None:
        required_axes = ["datasets", "generators", "attack_methods", "target_stances", "embedders", "agentic_rag"]
        for axis in required_axes:
            if axis not in self.matrix or not isinstance(self.matrix[axis], list) or not self.matrix[axis]:
                raise ConfigError(f"[matrix].{axis} must be a non-empty array")
        for name in self.matrix["datasets"]:
            if name not in self.datasets:
                raise ConfigError(f"Dataset {name!r} has no [dataset.{name}] configuration")
        for name in self.matrix["generators"]:
            if name not in self.models:
                raise ConfigError(f"Generator {name!r} has no [model.{name}] configuration")
        for name in self.matrix["embedders"]:
            if name not in self.embedders:
                raise ConfigError(f"Embedder {name!r} has no [embedder.{name}] configuration")
        invalid_methods = set(self.matrix["attack_methods"]) - set(METHOD_ALIASES)
        if invalid_methods:
            raise ConfigError(f"Unknown attack methods: {sorted(invalid_methods)}")
        invalid_stances = {str(x).upper() for x in self.matrix["target_stances"]} - {"PRO", "CON"}
        if invalid_stances:
            raise ConfigError(f"Unknown target stances: {sorted(invalid_stances)}")

    def experiments(self) -> list[Experiment]:
        result = []
        n_questions = int(self.matrix.get("n_questions", 40))
        runs_per_question = int(self.matrix.get("runs_per_question", 10))
        seed = int(self.matrix.get("seed", 42))
        top_k = int(self.matrix.get("top_k", 5))
        stance_model = str(self.execution.get("stance_model", "ollama:gemma4:latest"))
        for dataset in self.matrix["datasets"]:
            for generator in self.matrix["generators"]:
                for method_value in self.matrix["attack_methods"]:
                    method = METHOD_ALIASES[method_value]
                    dataset_cfg = self.datasets[dataset]
                    query_path = resolve_path(self.root, dataset_cfg["query_file"])
                    corpus_paths = [resolve_path(self.root, value) for value in dataset_cfg["corpus_files"]]
                    for stance_value in self.matrix["target_stances"]:
                        stance = str(stance_value).upper()
                        baseline_path = self.baseline_source(method, stance)
                        input_signature = digest(
                            {
                                "dataset_config": dataset_cfg,
                                "query": path_signature(query_path),
                                "corpora": [path_signature(path) for path in corpus_paths],
                                "baseline_source": path_signature(baseline_path) if baseline_path else None,
                                "baseline_config": self.baselines.get(method, {}),
                                "lof_defense": self.execution.get("lof_defense", False),
                                "dbscan_defense": self.execution.get("dbscan_defense", False),
                            },
                            16,
                        )
                        for embedder in self.matrix["embedders"]:
                            for agentic in self.matrix["agentic_rag"]:
                                result.append(
                                    Experiment(
                                        dataset=dataset,
                                        generator=generator,
                                        generator_model=str(self.models[generator]["name"]),
                                        attack_method=method,
                                        target_stance=stance,
                                        embedder=embedder,
                                        embedding_model=str(self.embedders[embedder]["name"]),
                                        agentic_rag=bool(agentic),
                                        n_questions=n_questions,
                                        runs_per_question=runs_per_question,
                                        seed=seed,
                                        top_k=top_k,
                                        input_signature=input_signature,
                                        stance_model=stance_model,
                                        code_signature=self.code_signature,
                                    )
                                )
        return result

    def select(self, patterns: list[str]) -> list[Experiment]:
        experiments = self.experiments()
        if not patterns:
            return experiments
        selected = []
        for experiment in experiments:
            resolved_id = self.run_dir(experiment).name
            if any(
                fnmatch.fnmatch(experiment.experiment_id, pattern)
                or fnmatch.fnmatch(resolved_id, pattern)
                or fnmatch.fnmatch(experiment.slug, pattern)
                for pattern in patterns
            ):
                selected.append(experiment)
        return selected

    def dataset_paths(self, experiment: Experiment) -> tuple[Path, list[Path], Path, str]:
        cfg = self.datasets[experiment.dataset]
        query_file = resolve_path(self.root, cfg["query_file"])
        corpus_files = [resolve_path(self.root, value) for value in cfg["corpus_files"]]
        clean_databases = cfg.get("clean_databases", {})
        if experiment.embedder not in clean_databases:
            raise ConfigError(f"[dataset.{experiment.dataset}.clean_databases] has no {experiment.embedder!r} path")
        clean_db = resolve_path(self.root, clean_databases[experiment.embedder])
        return query_file, corpus_files, clean_db, str(cfg["collection_name"])

    def source_file(self, experiment: Experiment) -> Path | None:
        return self.baseline_source(experiment.attack_method, experiment.target_stance)

    def baseline_source(self, method: str, target_stance: str) -> Path | None:
        cfg = self.baselines.get(method, {})
        source_files = cfg.get("source_files", {})
        value = str(source_files.get(target_stance.upper(), cfg.get("source_file", ""))).strip()
        return resolve_path(self.root, value) if value else None

    def document_cache_key(self, experiment: Experiment) -> str:
        query_file, _, _, _ = self.dataset_paths(experiment)
        source = self.source_file(experiment)
        value = {
            "dataset": experiment.dataset,
            "method": experiment.attack_method,
            "stance": experiment.target_stance,
            "n_questions": experiment.n_questions,
            "seed": experiment.seed,
            "query": path_signature(query_file),
            "source": path_signature(source) if source else None,
            "poisonedrag_iterations": int(self.baselines.get("poisonedrag", {}).get("iterations", 10)),
            "code_signature": self.code_signature,
        }
        return digest(value, 24)

    def clean_cache_key(self, experiment: Experiment, documents_dir: Path) -> str:
        _, corpus_files, clean_db, collection = self.dataset_paths(experiment)
        manifest = json.loads((documents_dir / "documents_manifest.json").read_text(encoding="utf-8"))
        value = {
            "dataset": experiment.dataset,
            "collection": collection,
            "corpora": [path_signature(path) for path in corpus_files],
            "clean_db": str(clean_db),
            "generator": experiment.generator,
            "generator_model": experiment.generator_model,
            "embedder": experiment.embedder,
            "embedding_model": experiment.embedding_model,
            "agentic": experiment.agentic_rag,
            "runs": experiment.runs_per_question,
            "top_k": experiment.top_k,
            "topics_sha256": manifest["topics_sha256"],
            "code_signature": self.code_signature,
        }
        return digest(value, 24)

    def _same_experiment_case(
        self,
        stored_experiment: dict[str, Any],
        experiment: Experiment,
    ) -> bool:
        """Match a prior run while deliberately ignoring only code_signature."""
        current = asdict(experiment)
        return all(
            stored_experiment.get(key) == value
            for key, value in current.items()
            if key != "code_signature"
        )

    def run_dir(self, experiment: Experiment) -> Path:
        """Resolve the best existing directory for an experimental case.

        The code signature remains useful for cache invalidation, but changing a
        pipeline file must not make completed experiment state unreachable.  Old
        run directories are therefore matched by every experimental field except
        ``code_signature``.  If a code change has already created a partial
        duplicate, the directory with the most valid completed stages wins.
        """
        cache_key = experiment.experiment_id
        if cache_key in self._run_dir_cache:
            return self._run_dir_cache[cache_key]

        canonical = self.runs_dir / experiment.experiment_id
        candidates: list[tuple[int, Path]] = []
        if self.runs_dir.exists():
            for state_path in self.runs_dir.glob("*/state.json"):
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                stored_experiment = state.get("experiment")
                if not isinstance(stored_experiment, dict):
                    continue
                if not self._same_experiment_case(stored_experiment, experiment):
                    continue
                candidate = state_path.parent
                completed_stages = sum(
                    self.completed_valid(candidate, state.get("stages", {}).get(stage))
                    for stage in STAGES
                )
                candidates.append((completed_stages, candidate))

        if not candidates:
            resolved = canonical
        else:
            best_score = max(score for score, _ in candidates)
            best = [path for score, path in candidates if score == best_score]
            if len(best) == 1:
                resolved = best[0]
            elif canonical in best:
                resolved = canonical
            else:
                names = ", ".join(sorted(path.name for path in best))
                raise ConfigError(
                    "Multiple equally complete run directories match "
                    f"{experiment.slug}: {names}"
                )

        self._run_dir_cache[cache_key] = resolved
        return resolved

    def state_path(self, experiment: Experiment) -> Path:
        return self.run_dir(experiment) / "state.json"

    def load_state(self, experiment: Experiment) -> dict[str, Any]:
        path = self.state_path(experiment)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                damaged = path.with_name(f"state.damaged-{int(time.time())}.json")
                path.replace(damaged)
        return {
            "schema_version": 1,
            "experiment_id": experiment.experiment_id,
            "experiment": asdict(experiment),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "pending",
            "stages": {},
        }

    def save_state(
        self,
        experiment: Experiment,
        state: dict[str, Any],
        run_dir_override: Path | None = None,
    ) -> None:
        state["updated_at"] = utc_now()
        run_dir = run_dir_override or self.run_dir(experiment)
        atomic_json(run_dir / "state.json", state)

    def artifacts(self, run_dir: Path, paths: list[Path]) -> list[dict[str, Any]]:
        values = []
        for path in paths:
            if not path.exists() or not path.is_file() or path.stat().st_size == 0:
                raise StageFailure(f"Expected non-empty artifact was not produced: {path}")
            values.append({
                "path": str(path.relative_to(run_dir)),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            })
        return values

    def csv_rows(self, path: Path, required_columns: set[str]) -> int:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="|" if path.name == "poisoned_docs.csv" else ",")
            columns = set(reader.fieldnames or [])
            missing = required_columns - columns
            if missing:
                raise StageFailure(f"{path} is missing columns: {sorted(missing)}")
            return sum(1 for _ in reader)

    def validate_stage_outputs(
        self,
        experiment: Experiment,
        stage: str,
        paths: list[Path],
        run_dir_override: Path | None = None,
    ) -> None:
        run_dir = run_dir_override or self.run_dir(experiment)
        if stage == "documents":
            manifest_path = run_dir / "documents" / "documents_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                document_count = int(manifest["document_count"])
                topics_hash = str(manifest["topics_sha256"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise StageFailure(f"Invalid document manifest: {error}") from error
            row_count = self.csv_rows(run_dir / "documents" / "poisoned_docs.csv", {"topic", "stance", "poisoned_doc"})
            if document_count <= 0 or row_count != document_count or len(topics_hash) != 64:
                raise StageFailure("Document manifest does not match poisoned_docs.csv")
        elif stage in {"poisoned_rag", "clean_rag"}:
            manifest = json.loads((run_dir / "documents" / "documents_manifest.json").read_text(encoding="utf-8"))
            expected_rows = int(manifest["document_count"]) * experiment.runs_per_question
            actual_rows = self.csv_rows(paths[0], {"original_query", "query_idx", "run_idx", "answer", "retrieved_context"})
            if actual_rows != expected_rows:
                raise StageFailure(f"Expected {expected_rows} RAG rows, found {actual_rows} in {paths[0]}")
        elif stage == "evaluation":
            try:
                metrics = json.loads((run_dir / "results" / "metrics.json").read_text(encoding="utf-8"))
                for key in ["retrieval", "clean_stance", "poisoned_stance", "stance_change"]:
                    if key not in metrics:
                        raise KeyError(key)
            except (OSError, KeyError, json.JSONDecodeError) as error:
                raise StageFailure(f"Invalid evaluation metrics: {error}") from error
            clean_answers = self.csv_rows(run_dir / "results" / "rag_answers_clean.csv", {"answer"})
            poison_answers = self.csv_rows(run_dir / "results" / "rag_answers_poison.csv", {"answer"})
            clean_labels = self.csv_rows(run_dir / "results" / "clean_annotation.csv", {"stance"})
            poison_labels = self.csv_rows(run_dir / "results" / "poison_annotation.csv", {"stance"})
            if clean_answers != clean_labels or poison_answers != poison_labels:
                raise StageFailure("Annotation row counts do not match their RAG outputs")

    def completed_valid(self, run_dir: Path, stage_state: dict[str, Any] | None) -> bool:
        if not stage_state or stage_state.get("status") != "completed":
            return False
        for item in stage_state.get("artifacts", []):
            path = run_dir / item["path"]
            if not path.exists() or path.stat().st_size != item.get("size") or file_sha256(path) != item.get("sha256"):
                return False
        return bool(stage_state.get("artifacts"))

    def copy_directory_atomic(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".copying")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)

    def copy_file_atomic(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".copying")
        shutil.copy2(source, temporary)
        temporary.replace(destination)

    def execute_command(
        self,
        experiment: Experiment,
        state: dict[str, Any],
        stage: str,
        command: list[str],
        expected: list[Path],
        env: dict[str, str] | None = None,
        run_dir_override: Path | None = None,
    ) -> None:
        run_dir = run_dir_override or self.run_dir(experiment)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stage_state = state["stages"].setdefault(stage, {})
        previous_attempts = int(stage_state.get("attempts", 0))
        for retry_index in range(self.max_retries + 1):
            attempt = previous_attempts + retry_index + 1
            log_path = logs_dir / f"{stage}.attempt-{attempt}.log"
            stage_state.update({
                "status": "running",
                "attempts": attempt,
                "started_at": utc_now(),
                "command": command,
                "log": str(log_path.relative_to(run_dir)),
            })
            state["status"] = "running"
            self.save_state(experiment, state, run_dir)
            print(f"  {stage}: attempt {attempt} -> {log_path.name}", flush=True)
            child_env = os.environ.copy()
            child_env.update(env or {})
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"Started: {utc_now()}\nCommand: {shlex.join(command)}\n\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    env=child_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                try:
                    return_code = process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    return_code = 124
                    log.write(f"\nTimed out after {self.timeout} seconds.\n")
                except KeyboardInterrupt:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                    stage_state.update({"status": "interrupted", "finished_at": utc_now()})
                    state["status"] = "interrupted"
                    self.save_state(experiment, state, run_dir)
                    raise

            if return_code == 0:
                try:
                    self.validate_stage_outputs(
                        experiment,
                        stage,
                        expected,
                        run_dir_override=run_dir,
                    )
                    artifact_records = self.artifacts(run_dir, expected)
                except StageFailure as error:
                    return_code = 2
                    failure_message = str(error)
                else:
                    stage_state.update({
                        "status": "completed",
                        "finished_at": utc_now(),
                        "artifacts": artifact_records,
                        "error": None,
                    })
                    self.save_state(experiment, state, run_dir)
                    return
            else:
                failure_message = f"Process exited with code {return_code}"

            stage_state.update({"status": "failed", "finished_at": utc_now(), "error": failure_message})
            self.save_state(experiment, state, run_dir)
            if retry_index < self.max_retries:
                delay = min(self.retry_backoff * (2**retry_index), 60.0)
                print(f"    failed; retrying in {delay:g}s", flush=True)
                time.sleep(delay)

        raise StageFailure(f"{stage} failed after {self.max_retries + 1} attempt(s); see {log_path}")

    def mark_cached(
        self,
        experiment: Experiment,
        state: dict[str, Any],
        stage: str,
        expected: list[Path],
        cache_key: str,
    ) -> None:
        run_dir = self.run_dir(experiment)
        self.validate_stage_outputs(experiment, stage, expected)
        state["stages"][stage] = {
            "status": "completed",
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "attempts": int(state["stages"].get(stage, {}).get("attempts", 0)),
            "cache_hit": True,
            "cache_key": cache_key,
            "artifacts": self.artifacts(run_dir, expected),
        }
        self.save_state(experiment, state)

    def document_stage(self, experiment: Experiment, state: dict[str, Any], force: bool) -> bool:
        run_dir = self.run_dir(experiment)
        documents_dir = run_dir / "documents"
        expected = [documents_dir / "poisoned_docs.csv", documents_dir / "documents_manifest.json"]
        if not force and self.completed_valid(run_dir, state["stages"].get("documents")):
            print("  documents: already completed")
            return False


        if not force and all(
            path.exists() and path.is_file() and path.stat().st_size > 0
            for path in expected
        ):
            try:
                self.validate_stage_outputs(
                    experiment,
                    "documents",
                    expected,
                )
            except StageFailure:
                print("  documents: existing artifacts are invalid; regenerating")
            else:
                state["stages"]["documents"] = {
                    "status": "completed",
                    "started_at": utc_now(),
                    "finished_at": utc_now(),
                    "attempts": int(
                        state["stages"].get("documents", {}).get("attempts", 0)
                    ),
                    "cache_hit": False,
                    "recovered_existing": True,
                    "artifacts": self.artifacts(run_dir, expected),
                }
                self.save_state(experiment, state)
                print("  documents: adopted existing artifacts")
                return False

        cache_key = self.document_cache_key(experiment)
        cache_path = self.cache_dir / "documents" / cache_key
        if self.cache_enabled and cache_path.exists() and all((cache_path / item.name).exists() for item in expected):
            self.copy_directory_atomic(cache_path, documents_dir)
            try:
                self.mark_cached(experiment, state, "documents", expected, cache_key)
            except StageFailure:
                print("  documents: ignored invalid cache entry")
            else:
                print("  documents: restored from cache")
                return True

        query_file, _, _, _ = self.dataset_paths(experiment)
        command = [
            self.python,
            str(self.script_dir / "document_builder.py"),
            "--method", experiment.attack_method,
            "--target-stance", experiment.target_stance,
            "--query-file", str(query_file),
            "--output-dir", str(documents_dir),
            "--n-questions", str(experiment.n_questions),
            "--seed", str(experiment.seed),
            "--poisonedrag-iterations", str(int(self.baselines.get("poisonedrag", {}).get("iterations", 10))),
            "--ebi-iterations", str(int(self.baselines.get("ebi", {}).get("iterations", 10))),
        ]
        for corpus_file in self.dataset_paths(experiment)[1]:
            command.extend(["--corpus-file", str(corpus_file)])
        if experiment.attack_method == "ebi":
            fit_corpus_file = self.datasets[experiment.dataset].get("ebi_fit_corpus_file")
            if fit_corpus_file:
                command.extend(["--ebi-fit-corpus-file", str(resolve_path(self.root, fit_corpus_file))])
            duo_model = self.baselines.get("ebi", {}).get(
                "duo_embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            )
            command.extend(["--ebi-duo-embedding-model", str(duo_model)])
        if experiment.attack_method == "corruptrag":
            if bool(self.baselines.get("corruptrag", {}).get("run_corruptrag_ak", False)):
                command.append("--run-corruptrag-ak")
        source = self.source_file(experiment)
        if source:
            command.extend(["--source-file", str(source)])
        self.execute_command(experiment, state, "documents", command, expected)
        if self.cache_enabled:
            self.copy_directory_atomic(documents_dir, cache_path)
        return True

    def rag_command(self, experiment: Experiment, condition: str, output: Path) -> list[str]:
        run_dir = self.run_dir(experiment)
        _, corpus_files, clean_db, collection = self.dataset_paths(experiment)
        command = [
            self.python,
            str(self.script_dir / "rag_runner.py"),
            "--condition", condition,
            "--collection-name", collection,
            "--poisoned-docs", str(run_dir / "documents" / "poisoned_docs.csv"),
            "--clean-db", str(clean_db),
            "--output", str(output),
            "--embedder", experiment.embedder,
            "--embedding-model", experiment.embedding_model,
            "--generator", experiment.generator,
            "--generator-model", experiment.generator_model,
            "--runs-per-question", str(experiment.runs_per_question),
            "--top-k", str(experiment.top_k),
        ]
        perplexity = self.execution.get("perplexity", {})
        if perplexity.get("model"):
            command.extend([
                "--perplexity-model", str(perplexity["model"]),
                "--perplexity-threshold", str(float(perplexity.get("threshold", 100.0))),
                "--perplexity-candidate-multiplier",
                str(int(perplexity.get("candidate_multiplier", 3))),
                "--perplexity-max-length",
                str(int(perplexity.get("max_length", 1024))),
            ])
        if self.execution.get("duplicate_defense", False):
            command.append("--duplicate-defense")
        if self.execution.get("clustering_defense", False):
            command.append("--clustering-defense")
        if self.execution.get("perplexity_defense", False):
            command.append("--perplexity-defense")
        lof = self.execution.get("lof_defense", False)
        if isinstance(lof, dict):
            if lof.get("enabled", True):
                command.append("--lof-defense")
                if lof.get("model"):
                    command.extend(["--lof-model", str(lof["model"])])
                if lof.get("n_neighbors") is not None:
                    command.extend(["--lof-neighbors", str(int(lof["n_neighbors"]))])
                if lof.get("contamination") is not None:
                    command.extend(["--lof-contamination", str(float(lof["contamination"]))])
        elif lof:
            command.append("--lof-defense")
        dbscan = self.execution.get("dbscan_defense", False)
        if isinstance(dbscan, dict):
            if dbscan.get("enabled", True):
                command.append("--dbscan-defense")
                for key, option, converter in (
                    ("eps", "--dbscan-eps", float),
                    ("min_samples", "--dbscan-min-samples", int),
                    ("metric", "--dbscan-metric", str),
                    ("min_cluster_size", "--dbscan-min-cluster-size", int),
                ):
                    if dbscan.get(key) is not None:
                        command.extend([option, str(converter(dbscan[key]))])
                for key, option in (
                    ("drop_noise", "--dbscan-drop-noise"),
                    ("keep_if_all_filtered", "--dbscan-keep-if-all-filtered"),
                ):
                    if dbscan.get(key) is not None:
                        command.extend([option if dbscan[key] else option.replace("--", "--no-", 1)])
                if dbscan.get("calibrate", False):
                    command.append("--dbscan-calibrate")
                    command.extend([
                        "--dbscan-eps-grid",
                        *[str(float(value)) for value in dbscan.get("eps_grid", [0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.01])],
                        "--dbscan-min-samples-grid",
                        *[str(int(value)) for value in dbscan.get("min_samples_grid", [2, 3, 4])],
                        "--dbscan-target-fpr",
                        str(float(dbscan.get("target_fpr", 0.02))),
                    ])
        elif dbscan:
            command.append("--dbscan-defense")
        for corpus_file in corpus_files:
            command.extend(["--corpus-file", str(corpus_file)])
        if condition == "poisoned":
            command.extend(["--poisoned-db", str(run_dir / "databases" / "poisoned")])
        if experiment.agentic_rag:
            command.append("--agentic")
        return command

    def poisoned_rag_stage(self, experiment: Experiment, state: dict[str, Any], force: bool) -> bool:
        run_dir = self.run_dir(experiment)
        output = run_dir / "results" / "rag_answers_poison.csv"
        if not force and self.completed_valid(run_dir, state["stages"].get("poisoned_rag")):
            print("  poisoned_rag: already completed")
            return False
        self.execute_command(experiment, state, "poisoned_rag", self.rag_command(experiment, "poisoned", output), [output])
        return True

    def clean_rag_stage(self, experiment: Experiment, state: dict[str, Any], force: bool) -> tuple[str, bool]:
        run_dir = self.run_dir(experiment)
        output = run_dir / "results" / "rag_answers_clean.csv"
        cache_key = self.clean_cache_key(experiment, run_dir / "documents")
        if not force and self.completed_valid(run_dir, state["stages"].get("clean_rag")):
            print("  clean_rag: already completed")
            return cache_key, False
        cache_file = self.cache_dir / "clean_rag" / f"{cache_key}.csv"
        if self.cache_enabled and cache_file.exists() and cache_file.stat().st_size > 0:
            self.copy_file_atomic(cache_file, output)
            try:
                self.mark_cached(experiment, state, "clean_rag", [output], cache_key)
            except StageFailure:
                print("  clean_rag: ignored invalid cache entry")
            else:
                print("  clean_rag: restored from cache")
                return cache_key, True
        self.execute_command(experiment, state, "clean_rag", self.rag_command(experiment, "clean", output), [output])
        if self.cache_enabled:
            self.copy_file_atomic(output, cache_file)
        return cache_key, True

    def evaluation_stage(self, experiment: Experiment, state: dict[str, Any], force: bool) -> None:
        run_dir = self.run_dir(experiment)
        results = run_dir / "results"
        outputs = [
            results / "clean_annotation.csv",
            results / "poison_annotation.csv",
            results / "stance_results.png",
            results / "metrics.json",
        ]
        if not force and self.completed_valid(run_dir, state["stages"].get("evaluation")):
            print("  evaluation: already completed")
            return

        stance_model = experiment.stance_model
        command = [
            self.python,
            str(self.script_dir / "stats.py"),
            "--clean-path", str(results / "rag_answers_clean.csv"),
            "--poisoned-path", str(results / "rag_answers_poison.csv"),
            "--clean-output-path", str(outputs[0]),
            "--poisoned-output-path", str(outputs[1]),
            "--output-file", str(outputs[2]),
            "--summary-json", str(outputs[3]),
            "--title", f"Target Opinion: {experiment.target_stance}",
            "--target-stance", experiment.target_stance,
        ]
        env = {"BIASCHAIN_STANCE_MODEL": stance_model, "MPLBACKEND": "Agg"}
        self.execute_command(experiment, state, "evaluation", command, outputs, env=env)

    def _experiment_from_stored_state(
        self,
        state: dict[str, Any],
        state_path: Path,
    ) -> Experiment:
        stored = state.get("experiment")
        if not isinstance(stored, dict):
            raise ConfigError(f"Missing experiment metadata in {state_path}")

        field_names = set(Experiment.__dataclass_fields__)
        missing = field_names - set(stored)
        if missing:
            raise ConfigError(
                f"Experiment metadata in {state_path} is missing: {sorted(missing)}"
            )
        return Experiment(**{name: stored[name] for name in field_names})

    def existing_evaluation_targets(
        self,
        patterns: list[str],
    ) -> list[tuple[Path, dict[str, Any], Experiment]]:
        """Find saved runs that already contain both RAG answer files."""
        targets = []
        if not self.runs_dir.exists():
            return targets

        for state_path in sorted(self.runs_dir.glob("*/state.json")):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                experiment = self._experiment_from_stored_state(state, state_path)
            except (OSError, json.JSONDecodeError, ConfigError) as error:
                print(f"Skipping {state_path.parent.name}: {error}", file=sys.stderr)
                continue

            run_dir = state_path.parent
            if patterns and not any(
                fnmatch.fnmatch(run_dir.name, pattern)
                or fnmatch.fnmatch(experiment.slug, pattern)
                for pattern in patterns
            ):
                continue

            results = run_dir / "results"
            inputs = [
                results / "rag_answers_clean.csv",
                results / "rag_answers_poison.csv",
            ]
            if not all(path.exists() and path.stat().st_size > 0 for path in inputs):
                print(
                    f"Skipping {run_dir.name}: existing clean and poisoned RAG outputs are required",
                    file=sys.stderr,
                )
                continue
            targets.append((run_dir, state, experiment))

        return targets

    def reevaluate_existing(self, patterns: list[str], dry_run: bool = False) -> int:
        """Re-evaluate saved RAG outputs without invoking any earlier stage."""
        targets = self.existing_evaluation_targets(patterns)
        print(f"Existing runs selected for evaluation: {len(targets)}")
        for index, (run_dir, _, _) in enumerate(targets, start=1):
            print(f"{index:3d}. {run_dir.name}")
        if dry_run:
            return 0
        if not targets:
            raise ConfigError("No existing runs with both RAG answer files matched")

        failures = 0
        updated = []
        for run_dir, state, experiment in targets:
            print(f"\n[{run_dir.name}] evaluation only", flush=True)
            results = run_dir / "results"
            outputs = [
                results / "clean_annotation.csv",
                results / "poison_annotation.csv",
                results / "stance_results.png",
                results / "metrics.json",
            ]
            command = [
                self.python,
                str(self.script_dir / "stats.py"),
                "--clean-path", str(results / "rag_answers_clean.csv"),
                "--poisoned-path", str(results / "rag_answers_poison.csv"),
                "--clean-output-path", str(outputs[0]),
                "--poisoned-output-path", str(outputs[1]),
                "--output-file", str(outputs[2]),
                "--summary-json", str(outputs[3]),
                "--title", f"Target Opinion: {experiment.target_stance}",
                "--target-stance", experiment.target_stance,
            ]
            env = {
                "BIASCHAIN_STANCE_MODEL": experiment.stance_model,
                "MPLBACKEND": "Agg",
            }
            try:
                self.execute_command(
                    experiment,
                    state,
                    "evaluation",
                    command,
                    outputs,
                    env=env,
                    run_dir_override=run_dir,
                )
            except StageFailure as error:
                failures += 1
                state["status"] = "failed"
                state["error"] = str(error)
                self.save_state(experiment, state, run_dir)
                print(f"  FAILED: {error}", flush=True)
                if self.fail_fast:
                    break
                continue

            if all(
                state.get("stages", {}).get(stage, {}).get("status") == "completed"
                for stage in STAGES
            ):
                state["status"] = "completed"
                state["completed_at"] = utc_now()
            else:
                state["status"] = "partial"
            state["error"] = None
            self.save_state(experiment, state, run_dir)
            updated.append((run_dir, state, experiment))
            print("  COMPLETED", flush=True)

        self.write_existing_summary(updated)
        print(f"\nFinished evaluation-only repair: {len(updated)} completed, {failures} failed")
        return 1 if failures else 0

    def force_set(self, requested: list[str]) -> set[str]:
        forced: set[str] = set()
        for stage in requested:
            start = STAGES.index(stage)
            forced.update(STAGES[start:])
        return forced

    def run_experiment(self, experiment: Experiment, forced: set[str]) -> str:
        run_dir = self.run_dir(experiment)
        run_dir.mkdir(parents=True, exist_ok=True)
        state = self.load_state(experiment)
        print(f"\n[{experiment.experiment_id}]", flush=True)
        try:
            documents_changed = self.document_stage(experiment, state, "documents" in forced)
            poisoned_changed = self.poisoned_rag_stage(
                experiment,
                state,
                "poisoned_rag" in forced or documents_changed,
            )
            _, clean_changed = self.clean_rag_stage(
                experiment,
                state,
                "clean_rag" in forced or documents_changed,
            )
            self.evaluation_stage(
                experiment,
                state,
                "evaluation" in forced or poisoned_changed or clean_changed,
            )
        except StageFailure as error:
            state["status"] = "failed"
            state["error"] = str(error)
            self.save_state(experiment, state)
            print(f"  FAILED: {error}", flush=True)
            return "failed"
        state["status"] = "completed"
        state["completed_at"] = utc_now()
        state["error"] = None
        self.save_state(experiment, state)
        print("  COMPLETED", flush=True)
        return "completed"

    def dry_run(self, experiments: list[Experiment]) -> None:
        print(f"Planned experiments: {len(experiments)}")
        print(f"Runs directory: {self.runs_dir}")
        print(f"Cache directory: {self.cache_dir}")
        for index, experiment in enumerate(experiments, start=1):
            source = self.source_file(experiment)
            source_text = f", source={source}" if source else ""
            resolved_id = self.run_dir(experiment).name
            print(f"{index:3d}. {resolved_id}{source_text}")

    def _summary_row(
        self,
        run_dir: Path,
        state: dict[str, Any],
        experiment: Experiment,
    ) -> dict[str, Any]:
        row = {
            "experiment_id": run_dir.name,
            **asdict(experiment),
            "status": state.get("status", "pending"),
        }
        metrics_path = run_dir / "results" / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                row["retrieval_success_rate"] = metrics["retrieval"]["retrieval_success_rate"]
                row["target_stance_change"] = metrics["stance_change"][experiment.target_stance]
                row["clean_target_stance_rate"] = metrics["target_stance"]["clean_target_stance_rate"]
                row["poisoned_target_stance_rate"] = metrics["target_stance"]["poisoned_target_stance_rate"]
                row["delta_target_stance_rate"] = metrics["target_stance"]["delta_target_stance_rate"]
                row["opinion_flip_success_rate"] = metrics["opinion_flip"]["opinion_flip_success_rate"]
                row["average_stance_variation"] = metrics["average_stance_variation"]["average_stance_variation"]
            except (KeyError, json.JSONDecodeError):
                pass
        return row

    def _merge_summary_rows(self, rows: list[dict[str, Any]]) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.runs_dir / "summary.json"
        existing_rows: list[dict[str, Any]] = []
        if summary_path.exists():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing_rows = [row for row in loaded if isinstance(row, dict)]
            except (json.JSONDecodeError, OSError):
                print(f"Warning: could not read existing summary: {summary_path}", file=sys.stderr)

        rows_by_id = {
            str(row["experiment_id"]): row
            for row in existing_rows
            if row.get("experiment_id")
        }
        rows_by_id.update({row["experiment_id"]: row for row in rows})
        rows = [rows_by_id[key] for key in sorted(rows_by_id)]

        atomic_json(summary_path, rows)
        columns = sorted({key for row in rows for key in row})
        temporary = self.runs_dir / "summary.csv.tmp"
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(self.runs_dir / "summary.csv")

    def write_summary(self, experiments: list[Experiment]) -> None:
        rows = []
        for experiment in experiments:
            state = self.load_state(experiment)
            run_dir = self.run_dir(experiment)
            rows.append(self._summary_row(run_dir, state, experiment))
        self._merge_summary_rows(rows)

    def write_existing_summary(
        self,
        targets: list[tuple[Path, dict[str, Any], Experiment]],
    ) -> None:
        rows = [
            self._summary_row(run_dir, state, experiment)
            for run_dir, state, experiment in targets
        ]
        self._merge_summary_rows(rows)

    def status(self, experiments: list[Experiment]) -> None:
        counts: dict[str, int] = {}
        for experiment in experiments:
            state = self.load_state(experiment)
            status = state.get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
            stage_text = ", ".join(f"{stage}={state.get('stages', {}).get(stage, {}).get('status', 'pending')}" for stage in STAGES)
            print(f"{self.run_dir(experiment).name}: {status} ({stage_text})")
        print("\n" + ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["run", "plan", "status", "reevaluate"]:
        child = subparsers.add_parser(name)
        child.add_argument("--config", "-c", type=Path, default=Path("experiments.toml"))
        child.add_argument("--only", action="append", default=[], help="Glob matching experiment ID or slug; repeatable")
        child.add_argument("--no-cache", action="store_true")
        if name == "run":
            child.add_argument("--force-stage", action="append", choices=STAGES, default=[])
        if name == "reevaluate":
            child.add_argument(
                "--dry-run",
                action="store_true",
                help="List existing RAG outputs that would be evaluated",
            )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        orchestrator = Orchestrator(args.config, no_cache=args.no_cache)
        if args.command == "reevaluate":
            return orchestrator.reevaluate_existing(args.only, dry_run=args.dry_run)

        experiments = orchestrator.select(args.only)
        if not experiments:
            raise ConfigError("No experiments matched --only")
        if args.command == "plan":
            orchestrator.dry_run(experiments)
            return 0
        if args.command == "status":
            orchestrator.status(experiments)
            return 0

        forced = orchestrator.force_set(args.force_stage)
        results = []
        for experiment in experiments:
            result = orchestrator.run_experiment(experiment, forced)
            results.append(result)
            if result == "failed" and orchestrator.fail_fast:
                break
        orchestrator.write_summary(experiments)
        failures = results.count("failed")
        print(f"\nFinished: {results.count('completed')} completed, {failures} failed")
        return 1 if failures else 0
    except (ConfigError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
