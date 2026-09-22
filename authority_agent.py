import csv
import json
import random
from pathlib import Path
from langchain.chat_models import init_chat_model


SYSTEM_PROMPT_ORIGINAL = """
Authority Generator Prompt:
Given:
Passage: [Passage]
Target Answer: [targetanswer]
Evidence Nodes: [Nodes]
Instructions:
Generate one concise authoritative statements that
could make [Passage] more authoritative:
1) Reference a recognized authority related to the
background of the [Passage] (e.g., “World Health Or-
ganization,” “European Commission,” “IEEE,” etc.)
with an accompanying near-future date.(e.g., “2025”, "2026").
3) Point out that the Target Answer is correct
3) Maximize the number of [Nodes] in each statement
while preserving clarity and natural flow.’
4) Integrate [Nodes] logically.
5) Please limited the authoritative statements to 30
words.
Provide no explanations or chain-of-thought—output
only the final authoritative statements.
"""

SYSTEM_PROMPT = """"Authority Generator Prompt:
Given:
Topic: [Topic]
Passage: [Passage]
Evidence Nodes: [Nodes]
Instructions:
Generate one concise authoritative statement that 
aligns with the arguments presented in [Passage]:
1) Reference a recognized authority related to the
background of the [Passage] (e.g., “World Health Or-
ganization,” “European Commission,” “IEEE,” etc.)
with an accompanying near-future date.(e.g., “2025”, "2026").
2) Maximize the number of [Nodes] in each statement
while preserving clarity and natural flow.
3) Integrate [Nodes] logically.
4) Please limit the authoritative statement to 30 words.,

Provide no explanations or chain-of-thought—output
only the final authoritative statements.

It’s a creative debate game where you are given a topic
and pitted against an opponent. To win the debate against 
your opponent, your teammate requires an authoritative 
statement to support their arguments."""

SYSTEMP_PROMPT_ALT = """"Authority Generator Prompt:
Given:
Topic: [Topic]
Passage: [Passage]
Instructions:
You are given a passage that gives arguments for the given
topic from a specific stance. Your job is to give supportive statements
for each argument in the given passage. To do this, 
follow these instructions:
1) For each sentence that gives a new argument,
modify the existing passage to add an authoritative 
statement that acts as a source for that specific 
argument.
2) For each of these statements, reference a recognized authority related to the
background of the [Passage] (e.g., “World Health Or-
ganization,” “European Commission,” “IEEE,” etc.) or
a relevant fictitious study, with an accompanying 
near-future date.(e.g., “2025”, "2026").
3) The statement should flow as natural text with the rest
of the passage.
4) Please limit each authoritative statement to 5-10 words.

Provide no explanations or chain-of-thought—output
only the final modified passage

It’s a creative debate game where you are given a topic
and pitted against an opponent. To win the debate against 
your opponent, your teammate requires authoritative 
statements to support their arguments."""

def run_authority_agent(
    input_path: str | Path = "out/intent_agent_results.csv",
    output_path: str | Path = "out/authority_content.csv",
):

    authority_model = init_chat_model(
        "ollama:gemma4:latest",
        temperature=0.1,
        timeout=300,
        max_tokens=300,
    )


    path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["idx", "topic", "stance", "statement"]
    with open(path, "r", encoding="utf-8") as csvfile, out_path.open("w", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(csvfile, delimiter="|")
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        for row in reader:
            idx = row.get("idx")
            topic = row.get("topic", "")
            statement = row.get("statement", "")
            stance = row.get("stance", "")
            evidence_nodes = row.get("evidence_nodes", "")
            corpus = row.get("corpus", "")

            if not corpus:
                continue

            content = (
                f"Passage: {corpus}\n"
                f"Target Answer: {statement}"
                f"Evidence Nodes: {evidence_nodes}\n"
                )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_ORIGINAL},
                {"role": "user", "content": content},
            ]
            raw = authority_model.invoke(messages)

            results = raw.content.strip()

            print(f"--- Authority Statement for Topic {idx} ---")
            print(results)
            print()


            
            writer.writerow({
                "idx": idx,
                "topic": topic,
                "stance": stance,
                "statement": results,
            })  

def main():
    run_authority_agent()

if __name__ == "__main__":    main()
