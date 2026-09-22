import json
from pathlib import Path

INPUT_FILE = Path("data/PROCON_data.json")
QUERY_OUT = Path("procon_queries.jsonl")
CORPUS_OUT = Path("procon_corpus.jsonl")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


with INPUT_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

query_rows = []
corpus_rows = []

for topic_idx, item in enumerate(data):
    topic = item["topic"]
    category = item.get("category", "")
    article_hash = topic_idx
    article_index = topic_idx

    # Queries
    for query_idx, query in enumerate(item["queries"]):
        query_id = f"{article_hash}_{query_idx}"

        query_rows.append({
            "_id": query_id,
            "text": query.strip() + " ",
            "metadata": {
                "article_hash": article_hash,
                "article_index": article_index,
                "query_index": query_idx,
                "topic": topic,
                "category": category,
            }
        })

    # Corpus passages
    for passage in item["passages"]:
        passage_id = passage[0]
        stance = passage[1]
        text = passage[3]

        corpus_rows.append({
            "_id": f"{article_hash}_{passage_id}",
            "text": text.strip(),
            "title": topic,
            "metadata": {
                "query_id": str(article_hash),
                "doc_id": str(passage_id),
                "stance": stance,
                "stance_label": "pro" if stance == 1 else "con",
                "topic": topic,
                "category": category,
                "description": "",
            }
        })

write_jsonl(QUERY_OUT, query_rows)
write_jsonl(CORPUS_OUT, corpus_rows)

print(f"Wrote {len(query_rows)} queries to {QUERY_OUT}")
print(f"Wrote {len(corpus_rows)} corpus rows to {CORPUS_OUT}")