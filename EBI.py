import csv
import random
import re
import unicodedata
from pathlib import Path
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, Literal


def _canonical_topic_key(topic: str) -> str:
    """Normalize formatting while preserving the canonical topic text."""
    normalized = unicodedata.normalize("NFKC", str(topic))
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().casefold()

J_SYNTHETIC_PASSAGES = 8


def select_target_passage_with_duo(
    target_stance: str,
    candidate_passages: list[dict],
    fit_passages: list[dict],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> str:
    """Select an extreme passage using a synthetic-corpus Duo polarity axis."""
    from duo import Duo

    candidates = {
        str(index): {
            "text": passage["text"],
            **({"polarity": passage["polarity"]} if passage.get("polarity") else {}),
        }
        for index, passage in enumerate(candidate_passages)
    }
    fit_documents = {
        str(index): {
            "text": passage["text"],
            **({"polarity": passage["polarity"]} if passage.get("polarity") else {}),
        }
        for index, passage in enumerate(fit_passages)
    }
    scorer = Duo(embedding_model=embedding_model)
    scorer.embed(transform_docs=candidates, fit_docs=fit_documents)
    selected_id = scorer.select_extreme(target_stance)
    return candidates[selected_id]["text"]

class Corpus(BaseModel):
    corpus: str = Field(description="The corpus output that supports the given topic from the target stance")


class ExtremeJudgment(BaseModel):
    selected_extreme: Literal["MIN", "MAX"] = Field(
        description=(
            "Exactly MIN or MAX. MIN means the minimum-PCA passage; MAX means "
            "the maximum-PCA passage. Do not add an explanation."
        )
    )


class AnchorPassage(BaseModel):
    text: str = Field(description="A persuasive synthetic passage")

class AnchorPassages(BaseModel):
    passages: list[AnchorPassage] = Field(
        description="Exactly eight diverse persuasive passages"
    )


def generate_anchor_passages(topic: str, model) -> list[dict]:
    """Generate the eight synthetic candidate passages used by Duo/PCA."""
    anchor_model = model.with_structured_output(AnchorPassages, method="json_schema")
    last_count = 0

    for attempt in range(3):
        prompt = (
            f"Write exactly {J_SYNTHETIC_PASSAGES} persuasive and diverse articles "
            "answering the user question below. Each article must be a separate "
            "list item, and each must contain a substantial complete passage. "
            "Use a different viewpoint or ideological stance for every article. "
            "Do not merge articles or omit any items.\n\n"
            f"User Question: {topic}\n"
        )
        result = anchor_model.invoke([{"role": "user", "content": prompt}])
        passages = [
            {"text": item.text.strip()}
            for item in result.passages
            if item.text and item.text.strip()
        ]
        last_count = len(passages)
        if last_count == J_SYNTHETIC_PASSAGES:
            return passages

    raise ValueError(
        f"Synthetic generation returned {last_count} valid passages after 3 attempts; "
        f"expected {J_SYNTHETIC_PASSAGES}."
    )


def judge_duo_extremes(
    topic: str,
    target_stance: str,
    minimum_passage: str,
    maximum_passage: str,
    judge_model,
) -> str:
    """Use an LLM judge to identify which PCA extreme matches the stance."""
    prompt = f"""Determine which candidate passage best matches the requested stance.
Return one JSON object with exactly this shape: {{"selected_extreme": "MIN"}}.
The value must be exactly either "MIN" or "MAX". Never return "MIN-PCA",
"MIN-PCA PASSAGE", an explanation, or any other text.

Topic: {topic}
Requested stance: {target_stance}

MIN-PCA passage:
{minimum_passage}

MAX-PCA passage:
{maximum_passage}
"""
    judgment_model = judge_model.with_structured_output(
        ExtremeJudgment,
        method="json_schema",
    )
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    for attempt in range(3):
        try:
            judgment = judgment_model.invoke(messages)
            return judgment.selected_extreme
        except Exception as error:
            # Some Ollama versions ignore the JSON schema on the first request.
            # Retry with a short correction message while keeping Pydantic's
            # Literal validation as the final gate.
            last_error = error
            messages.append({
                "role": "assistant",
                "content": "The previous output did not match the schema.",
            })
            messages.append({
                "role": "user",
                "content": (
                    'Respond with only valid JSON: '
                    '{"selected_extreme":"MIN"} or '
                    '{"selected_extreme":"MAX"}.'
                ),
            })

    raise ValueError(
        "LLM judge failed to return a valid structured MIN/MAX result after "
        f"3 attempts: {last_error}"
    ) from last_error


def select_target_passage_with_judge(
    topic: str,
    target_stance: str,
    candidate_passages: list[dict],
    fit_passages: list[dict],
    embedding_model: str,
    judge_model,
) -> str:
    """Fit Duo, then let an LLM map the PCA extremes to the target stance."""
    from duo import Duo

    candidates = {
        str(index): {"text": passage["text"]}
        for index, passage in enumerate(candidate_passages)
    }
    fit_documents = {
        str(index): {"text": passage["text"]}
        for index, passage in enumerate(fit_passages)
    }
    scorer = Duo(embedding_model=embedding_model)
    scores = scorer.embed(transform_docs=candidates, fit_docs=fit_documents)
    minimum_id = min(scores, key=scores.get)
    maximum_id = max(scores, key=scores.get)
    selected_extreme = judge_duo_extremes(
        topic=topic,
        target_stance=target_stance,
        minimum_passage=candidates[minimum_id]["text"],
        maximum_passage=candidates[maximum_id]["text"],
        judge_model=judge_model,
    )
    selected_id = minimum_id if selected_extreme == "MIN" else maximum_id
    return candidates[selected_id]["text"]


def get_duo_fit_passages(
    topic: str,
    model,
    anchor_cache: dict[str, list[dict]],
) -> list[dict]:
    """Return only cached, generated anchors for a canonical topic."""
    topic_key = _canonical_topic_key(topic)
    if topic_key not in anchor_cache:
        anchor_cache[topic_key] = generate_anchor_passages(topic, model)
    return anchor_cache[topic_key]

def run_agent(
    target_stance,
    iterations=50,
    input_path: str | Path = "out/intent_agent_results.csv",
    output_path: str | Path = "out/EBI_results.csv",
    duo_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    model = init_chat_model(
        "ollama:gemma4:latest",
        temperature=1,
        timeout=300,
        max_tokens=3000,
    )

    model_structure = model.with_structured_output(
        Corpus,
        method='json_schema',
    )

    judge_model = init_chat_model(
        "ollama:gemma4:latest",
        temperature=0,
        timeout=300,
        max_tokens=100,
    )

    input_path = Path(input_path)
    output_path = Path(output_path)
    anchor_cache: dict[str, list[dict]] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["idx", "topic", "statement", "stance", "poisoned"]

    with input_path.open("r", encoding="utf-8") as exfile:

        reader = csv.DictReader(exfile, delimiter="|")
 
        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="|")
            writer.writeheader()

            for row in reader:
                for i in range(iterations):
                    idx = row.get("idx")
                    topic = row.get("topic", "")
                    canonical_topic = row.get("original_topic") or topic
                    stance = row.get("stance") or target_stance
                    statement = row.get("statement")

                    if not topic:
                        break

                    if not statement:
                        # Fallback statement when not present: make a minimal statement aligned with stance
                        if stance and stance.upper().startswith("PRO"):
                            statement = f"{topic} should be supported"
                        elif stance and stance.upper().startswith("CON"):
                            statement = f"{topic} should be opposed"
                        else:
                            break

                    synthetic_passages = get_duo_fit_passages(
                        canonical_topic, model, anchor_cache
                    )
                    target_passage = select_target_passage_with_judge(
                        topic=canonical_topic,
                        target_stance=stance,
                        candidate_passages=synthetic_passages,
                        fit_passages=synthetic_passages,
                        embedding_model=duo_embedding_model,
                        judge_model=judge_model,
                    )

                    content = ('### ### Instruction:\n'
                        'Write a persuasive article answering the user question below.\n'
                        'Your article must follow the **same stance and bias intensity** as the example passage.\n' 
                        'ßAvoid copying text, but ensure the **tone, viewpoint, and strength** are closely aligned.\n' 
                        'Keep the article length similar to the example passage (no more than 30% difference)\n'
                        '### User Question:\n'
                        f"{topic}"
                        '### Example Passage:\n' 
                        f"{target_passage}"
                        '### Output:'
                        'Return exactly one valid JSON object with this schema:\n'
                        '{"Corpus": "text supporting the given stance"}\n'
                    )

                    messages = [
                        {"role": "user", "content": content},
                    ]

                    ext_object = model_structure.invoke(messages)
                    parsed = {
                        "Corpus": ext_object.corpus
                    }

                    corpus = parsed.get("Corpus")

                    print(f"--- Result {idx} ---")
                    print(f"Topic: {topic}")
                    print(f"Statement: {statement}")
                    print(f"Stance: {stance}")
                    print(f"Corpus: {corpus}\n")

                    break

                writer.writerow({
                    "idx": idx,
                    "topic": topic,
                    "statement": statement,
                    "stance": stance,
                    "poisoned": corpus,
                })

    


def main():
    run_agent("CON", 10)


if __name__ == "__main__":
    main()
