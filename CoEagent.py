import csv
import json
import random
from pathlib import Path
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List

SYSTEM_PROMPT_JUDGE = """CoE Judge Agent Prompt:
Given:
Passage: [Passage]
Evidence Nodes: [Nodes]
Evidence Relations: [Relations]
Intent: [Intent]
Topic: [Topic]
Statement: [Statement]
Check the [Passage] for Completeness:
1. Evidence Nodes Coverage
a) Does each sentence in [Passage] contain at least
one [Nodes]?
b) Does the [Passage] explicitly include all items
listed under [Nodes]?
c) Are there any cases where the keywords in [Passage]
are replaced by pronouns or vague synonyms (e.g.,
“it,” “they,” or “this” instead of the actual [Nodes])?
2. Evidence Relations Coverage (Skip if [Relations]
is empty)
a) Does the [Passage] clearly establish or infer all of
the provided [Relations]?
b) Are there any unclear or weakly supported relations
in [Passage]?
3. Intent Entailment
a) Can the specified [Intent] be found in or reasonably
inferred from the [Passage]?
Output Rules:
1) If all criteria are met (i.e., the Passage covers all
[Nodes], [Relations] if present, and [Intent]), output
only: Yes
2) If any criterion is not met:
Provide a set of revision suggestions for the [Passage].
Specifically:
a) Indicate how to add or replace missing keywords
(or remove ambiguous pronouns) in each sentence to
maximize the number of keywords.
b) Tell how to Revise or remove sentences that lack
keywords until each sentence contains at least one
keyword.
c) Explain how to clarify or insert any undefined or
weak relations (if [Relations] are given).
Do not output any step-by-step explanations or chain-
of-thought. Simply give "Yes" if all items are satisfied,
or directly provide the revision suggestions if not.
d) The revisions should not alter the intent of the passage.
The passage currently supports the statement made in [Statement].
Any modifications made HAS TO ENSURE THIS IS MAINTAINED. Do not
include any of your own thoughts or knowledge on the subject.
Assume that what is said in the passage is the absolute truth, 
while your job is only to ensure node and relationship coverage.
"""

SYSTEM_PROMPT_REVISION = """Revise Agent Prompt:
Given:
Passage: [Passage]
Advise: [Advise]
Instructions:
Incorporate any relevant suggestions from [Advise]
into [Passage].
If there is any conflict between [Passage] and [Advise],
[Advise] takes priority.
Output:
The revised [Passage], fully updated according to
[Advise].
Please limited the revised [Passage] to 100 words.
No explanations or step-by-step reasoning only the
final revised text."""


SYSTEM_PROMPT_EVIDENCE_REL = """Evidence Relations Extraction Prompt:
Extract evidence relations from the input topic and evidence nodes. Requirements: 1) Each relation contains two
elements: implied evidence nodes and relation description 2) Relation descriptions only involve the two connected nodes
3) Skip if no relation exists between nodes
Output must be in JSON format. Examples:
E1: Q: 750 7th Avenue and 101 Park Avenue, are located in which city? Nodes: ["750 7th Avenue", "101 Park Avenue"]
Out: []
E2: Q: Lee Jun-fan played what character in "The Green Hornet" television series? Nodes: ["Lee Jun-fan", "The Green
Hornet"] Out: {"Evidence nodes":["Lee Jun-fan", "The Green Hornet"], "Evidence Relations": "played character in"}
E3: Q: In which stadium do the teams owned by Myra Kraft’s husband play? Nodes: ["teams", "Myra Kraft’s husband"]
Out: {"Evidence nodes":["teams", "Myra Kraft’s husband"], "Evidence Relations": "is owned by"]}
E4: Q: The Colts’ first ever draft pick was a halfback who won the Heisman Trophy in what year? Nodes: ["Colts’ first
ever draft pick", "halfback", "Heisman Trophy"] Out: {"Evidence nodes":["Colts’ first ever draft pick", "halfback"],
"Evidence Relations": "was"}
E5: Q: The Golden Globe Award winner for best actor from "Roseanne" starred along what actress in Gigantic? Nodes:
["Golden Globe Award winner", "best actor", "Roseanne", "Gigantic"] Out: {"Evidence nodes":["Golden Globe Award
winner", "best actor"], "Evidence Relations": "for", "Evidence nodes":["best actor", "Roseanne"], "Evidence Relations":
"starred in"}
Question: [Question] Evidence nodes: [Evidence node] Output:"""

class EvidenceRelation(BaseModel):
    evidence_nodes: List[str] = Field(
        description="The evidence nodes involved in the relation"
    )
    evidence_relation: str = Field(
        description="The relation between the evidence nodes"
    )


class EvidenceRelationsOutput(BaseModel):
    relations: List[EvidenceRelation] = Field(
        description="All evidence relations found in the topic. Empty list if none exist."
    )

def run_coe_agent(
    input_path: str | Path = "out/intent_agent_results.csv",
    output_path: str | Path = "out/CoE_content.csv",
):

    judge_model = init_chat_model(
        "ollama:gemma4:latest",
        temperature=0,
        timeout=300,
        max_tokens=2000,
    )

    revise_model = init_chat_model(
        "ollama:gemma4:latest",
        temperature=0.1,
        timeout=300,
        max_tokens=500,
    )

    evidence_rel_model = init_chat_model(
        "ollama:gemma4:latest",
        temperature=0.1,
        timeout=300,
        max_tokens=300,
    )

    evidence_rel_model_structured = evidence_rel_model.with_structured_output(
        EvidenceRelationsOutput
    )

    path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_iter_count = 10
    fieldnames = ["idx", "topic", "stance", "corpus"]
    with open(path, "r", encoding="utf-8") as csvfile, out_path.open("w", newline="", encoding="utf-8") as outfile:
        reader = csv.DictReader(csvfile, delimiter="|")
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        for row in reader:
            count = 0
            idx = row.get("idx")
            topic = row.get("topic", "")
            statement = row.get("statement", "")
            stance = row.get("stance", "")
            intent = row.get("intent", "")
            evidence_nodes = row.get("evidence_nodes", "")
            corpus = row.get("corpus", "")
            

            if not corpus:
                continue

            evidence_relations = []

            rel_prompt = (
                f"Question: {topic}\n"
                f"Evidence nodes: {evidence_nodes}\n"
                "Extract all evidence relations. If no relations exist, return an empty relations list."
            )
            rel_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_EVIDENCE_REL},
                {"role": "user", "content": rel_prompt},
            ]
            rel_result = evidence_rel_model_structured.invoke(rel_messages)

            evidence_relations = [
                {
                    "Evidence nodes": rel.evidence_nodes,
                    "Evidence Relations": rel.evidence_relation,
                }
                for rel in rel_result.relations
            ]

            print("Relations:", evidence_relations)
            content = {}

            while count < max_iter_count:


                content = (
                    f"Passage: {corpus}\n"
                    f"Evidence Relations: {evidence_relations}"
                    f"Evidence Nodes: {evidence_nodes}\n"
                    f"Intent: {intent}\n"
                    f"Topic: {topic}\n"
                    f"Statement: {statement}"
                    "In your output, do not give any chain-of-thought where you go over the points separately." \
                    "Simply output the one word 'Yes' if all of them are met, or the specific revision suggestions for each sentence," \
                    "where you explicitly say the words that have to be changed in each sentence. " \
                    "Furthermore, do not just write the entire passage yourself"
                    
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_JUDGE},
                    {"role": "user", "content": content},
                ]
                judge_result = judge_model.invoke(messages)

                judge_raw = judge_result.content.strip()

                print(f"--- Judge Result for Topic {idx} ---")
                print(judge_raw)
                print()

                judge_text = judge_raw.lower()

                import re

                if re.search(r"\byes\b", judge_text):
                     break

                revision_advice = judge_raw

                revise_content = (
                    f"Passage: {corpus}\n"
                    f"Advise: {revision_advice}\n"
                    "Revise the Passage according to the Advise. "
                    "Do not include any explanations, reasoning, or notes in your response. "
                    "Include ONLY the revised passage in your response."
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_REVISION},
                    {"role": "user", "content": revise_content},
                ]

                revise_result = revise_model.invoke(messages)

                corpus = revise_result.content.strip()

                print(f"--- Revised corpus for {idx} ---")
                print(corpus)

                count += 1
            
            writer.writerow({
                "idx": idx,
                "topic": topic,
                "stance": stance,
                "corpus": corpus, 
            })  


def main():
    run_coe_agent()

if __name__ == "__main__":    main()
