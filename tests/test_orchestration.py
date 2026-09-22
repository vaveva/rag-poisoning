import json
import sys
import tempfile
import unittest
from pathlib import Path

from biaschain_orchestrator import Orchestrator
from document_builder import compose_poisoned_docs


CONFIG = b"""
[execution]
runs_dir = "runs"
cache_dir = "cache"
max_retries = 0

[matrix]
datasets = ["wiki_balance"]
generators = ["ollama"]
attack_methods = ["biaschain"]
target_stances = ["PRO", "CON"]
embedders = ["nomic"]
agentic_rag = [false, true]
n_questions = 40
runs_per_question = 10
seed = 42
top_k = 5

[dataset.wiki_balance]
query_file = "data/queries.jsonl"
corpus_files = ["data/corpus.jsonl"]
collection_name = "naturalcorpus"

[dataset.wiki_balance.clean_databases]
nomic = "db/nomic"

[model.ollama]
name = "llama3.1:8b"

[embedder.nomic]
name = "nomic-embed-text"
"""


class OrchestrationTests(unittest.TestCase):
    def make_orchestrator(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        config = root / "experiments.toml"
        config.write_bytes(CONFIG)
        return temporary, Orchestrator(config)

    def test_matrix_expansion_and_stable_ids(self):
        temporary, orchestrator = self.make_orchestrator()
        self.addCleanup(temporary.cleanup)
        experiments = orchestrator.experiments()
        self.assertEqual(4, len(experiments))
        self.assertEqual(len({item.experiment_id for item in experiments}), 4)
        self.assertEqual(
            [item.experiment_id for item in experiments],
            [item.experiment_id for item in orchestrator.experiments()],
        )

    def test_force_stage_invalidates_downstream(self):
        temporary, orchestrator = self.make_orchestrator()
        self.addCleanup(temporary.cleanup)
        self.assertEqual(
            {"poisoned_rag", "clean_rag", "evaluation"},
            orchestrator.force_set(["poisoned_rag"]),
        )

    def test_artifact_hash_controls_resume(self):
        temporary, orchestrator = self.make_orchestrator()
        self.addCleanup(temporary.cleanup)
        experiment = orchestrator.experiments()[0]
        run_dir = orchestrator.run_dir(experiment)
        artifact = run_dir / "results" / "value.txt"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("complete", encoding="utf-8")
        stage = {"status": "completed", "artifacts": orchestrator.artifacts(run_dir, [artifact])}
        self.assertTrue(orchestrator.completed_valid(run_dir, stage))
        artifact.write_text("changed", encoding="utf-8")
        self.assertFalse(orchestrator.completed_valid(run_dir, stage))

    def test_biaschain_document_composition(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "authority_content.csv").write_text(
                "idx|topic|stance|statement\n1|Question?|PRO|Authority\n", encoding="utf-8"
            )
            (output / "CoE_content.csv").write_text(
                "idx|topic|stance|corpus\n1|Question?|PRO|Evidence\n", encoding="utf-8"
            )
            path = compose_poisoned_docs("biaschain", output)
            self.assertTrue(path.exists())
            self.assertIn("Authority\nEvidence", path.read_text(encoding="utf-8"))
            manifest = json.loads((output / "documents_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(1, manifest["document_count"])

    def test_stage_retries_then_records_completion(self):
        temporary, orchestrator = self.make_orchestrator()
        self.addCleanup(temporary.cleanup)
        orchestrator.max_retries = 1
        orchestrator.retry_backoff = 0
        experiment = orchestrator.experiments()[0]
        run_dir = orchestrator.run_dir(experiment)
        counter = run_dir / "counter.txt"
        artifact = run_dir / "results" / "artifact.txt"
        state = orchestrator.load_state(experiment)
        code = (
            "from pathlib import Path\n"
            f"counter=Path({str(counter)!r})\n"
            f"artifact=Path({str(artifact)!r})\n"
            "counter.parent.mkdir(parents=True, exist_ok=True)\n"
            "value=int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(value + 1))\n"
            "if value == 0: raise SystemExit(7)\n"
            "artifact.parent.mkdir(parents=True, exist_ok=True)\n"
            "artifact.write_text('complete')\n"
        )
        orchestrator.execute_command(
            experiment,
            state,
            "test_stage",
            [sys.executable, "-c", code],
            [artifact],
        )
        self.assertEqual("completed", state["stages"]["test_stage"]["status"])
        self.assertEqual(2, state["stages"]["test_stage"]["attempts"])
        self.assertEqual("2", counter.read_text())


if __name__ == "__main__":
    unittest.main()
