# BiasChain experiment orchestration

The orchestrated path runs each experiment as four isolated, resumable stages:

1. build poisoned documents;
2. run poisoned RAG;
3. run or restore cached clean RAG;
4. classify both outputs and write metrics.

The orchestrator uses `document_builder.py` and the consolidated `rag_runner.py`, which avoid their shared-global-output and import-time database side effects.

## First run

Place this folder back at the root of the original project so that its `data/` files and Chroma directories match the paths in `experiments.toml`. Then activate the same Python environment used for the current pipeline.

```bash
python biaschain_orchestrator.py plan --config experiments.toml
python biaschain_orchestrator.py run --config experiments.toml
```

`experiments.toml` deliberately starts with only two experiments: WikiBalance, Ollama, BiasChain, nomic, non-agentic RAG, for both target stances. `experiments.full.toml` contains the 96 combinations of both datasets, both generators, all three attack methods, both stances, both embedders, and base/agentic RAG. Review that plan before starting because it is a very large and potentially costly run.

For OpenRouter runs, export the key before starting. The runner never prompts for a key, because a prompt would stall unattended execution.

```bash
export OPENROUTER_API_KEY="..."
python biaschain_orchestrator.py plan --config experiments.full.toml
python biaschain_orchestrator.py run --config experiments.full.toml
```

All Ollama generator, embedding, and stance-classifier models referenced by the chosen configuration must already be available to Ollama.

## Duplicate phrase filtering and semantic clustering

Running the duplicate phrase filtering or semantic clustering filtering defenses requires building new Chroma databases. If you have previously run experiments without the filtering defenses, the built databases need to be either deleted or renamed, so that the database building is triggered again.

To run the filtering defenses, use the following arguments in the `toml` file under the `[execution]` block:
- For duplicate phrase filtering, use `duplicate_defense = true`
- For semantic clustering filtering, use `clustering_defense = true`

## Perplexity filtering

Perplexity filtering is disabled by default. To enable it, install `torch` and `transformers`, then add this optional execution block:

```toml
[execution.perplexity]
model = "Qwen/Qwen2.5-0.5B"
threshold = 100.0
candidate_multiplier = 3
max_length = 1024
```

The runner retrieves `top_k * candidate_multiplier` documents, scores each document with the causal language model, removes documents whose perplexity exceeds `threshold`, and passes at most `top_k` documents to the generator. Lower perplexity means the text is more probable to the scoring model; it is a fluency/anomaly heuristic, not a relevance score. The same filter is applied to clean and poisoned RAG runs.

## Resume and inspect

Run the exact same `run` command again after an interruption. Completed stages whose artifacts still match their recorded hashes are skipped. Failed and interrupted stages are attempted again. No separate resume flag is needed.

```bash
python biaschain_orchestrator.py status --config experiments.toml
python biaschain_orchestrator.py run --config experiments.toml
```

Use `--only` with a glob to run or inspect part of the matrix:

```bash
python biaschain_orchestrator.py run -c experiments.full.toml --only '*procon*poisonedrag*con*qwen*base*'
```

To intentionally repeat a stage, use `--force-stage`. All downstream stages are also repeated so outputs cannot become inconsistent.

```bash
python biaschain_orchestrator.py run -c experiments.toml --force-stage poisoned_rag
```

## Files and failure behavior

Each experiment is stored under `runs/<experiment-id>/`:

- `documents/` contains the sampled topics and generated poison;
- `results/` contains clean/poisoned RAG CSVs, annotations, plot, and `metrics.json`;
- `logs/` contains a separate log for every stage attempt;
- `databases/poisoned/` is the experiment-local clean-database copy;
- `state.json` is the atomic resume record.

`runs/summary.csv` and `runs/summary.json` summarize the matrix. By default, a failed experiment is recorded and the next one continues. Set `fail_fast = true` only when you want the whole matrix to stop at the first failed experiment.

Each stage is retried according to `max_retries`. A value of `2` means one initial attempt plus two retries. Set `stage_timeout_minutes` to a positive value if a hung stage should be terminated automatically.

## Caching

Generated documents are cached across generator, embedder, and RAG-mode combinations when their dataset, attack, stance, seed, and source are identical. Clean RAG is cached using the actual topic-list hash plus dataset, corpus, generator, embedder, top-k, run count, and agentic mode. Clean stance annotations are cached separately by clean-result identity and classifier model.

Use `--no-cache` to bypass shared caches for a command. Per-experiment resume state still applies; combine it with `--force-stage documents` if every stage should truly be recomputed.

## Baseline topic sources

Baselines can sample the dataset's configured `query_file`, so an external source is no longer mandatory. To reproduce an old sample, set `source_file` under the relevant baseline section:

```toml
[baseline.poisonedrag]
source_file = "path/to/intent_agent_results.csv"
iterations = 10

[baseline.prompt_injection]
source_file = "path/to/prior_rag_answers.csv"
```

PoisonedRAG and FlippedRAG sources are normalized into the intent-agent schema. FlippedRAG additionally writes `FlippedRAG_imitation_pairs.json` containing its pseudo-relevant query/positive pairs. Prompt-injection sources may be either a RAG-answer CSV (`query_idx` and `rephrased_query`) or an intent-agent CSV (`idx` and `topic`).

If PRO and CON were sampled into separate files, configure both explicitly. A source containing a `stance` column is filtered to the current target stance and fails clearly when no matching rows exist.

```toml
[baseline.poisonedrag]
source_files = { PRO = "sources/pro_intents.csv", CON = "sources/con_intents.csv" }
iterations = 10
```

## Editing the matrix

The keys corresponding to the old flags are now:

| Old flag | Configuration / behavior |
| --- | --- |
| `BUILD_NEW_DOCS` | document stage cache plus `--force-stage documents` |
| `USE_POISONED_DB` | automatic poisoned and clean stages |
| `POISONED_DOC_METHOD` | `matrix.attack_methods` |
| `TARGET_STANCE` | `matrix.target_stances` |
| `EMBEDDER_NAME` | `matrix.embedders` |
| `USE_AGENT_RAG` | `matrix.agentic_rag` |
| `N_QUESTIONS` | `matrix.n_questions` |

The former two filename flags in `stats.py` are no longer used by orchestration. Every evaluation path is passed explicitly, and the statistics stage writes machine-readable `metrics.json`.
