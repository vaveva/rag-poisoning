#!/usr/bin/env python3
"""Inter-annotator agreement (Cohen's kappa) for two sampled_answers files.

Reads two annotation files (``.xlsx`` or ``.csv``), pairs their rows by
content, and computes Cohen's kappa on the annotation column (``Human
annotation`` if present, otherwise ``annotation``).

Rows are paired by a content key built from the provenance columns that
both files share (``run_id`` + ``query`` + ``answer``), so differing row
order between annotators is handled. Rows whose annotation is blank are
excluded from the kappa computation but reported.

Usage:
    python inter_annotator_agreement.py sampled_answers.xlsx "sampled_answers(2).xlsx"
    python inter_annotator_agreement.py a.xlsx b.xlsx --column "Human annotation"
    python inter_annotator_agreement.py a.xlsx b.xlsx --bootstrap 2000
    python inter_annotator_agreement.py a.xlsx b.xlsx --disagreements-out disagreements.csv
"""

import argparse
import csv
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

SSML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ANNOTATION_COLUMN_CANDIDATES = ["Human annotation", "annotation"]
KEY_COLUMNS = ["run_id", "query"]


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def col_letter_to_index(letters):
    """'A' -> 0, 'Z' -> 25, 'AA' -> 26, ..."""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def read_xlsx(path):
    """Read the first worksheet of an .xlsx file into a list of dicts."""
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())

        # Shared strings (optional part of the format).
        shared = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{SSML}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{SSML}t")))

        # Locate the first sheet via workbook.xml + its relationships,
        # falling back to sheet1.xml.
        sheet_target = "xl/worksheets/sheet1.xml"
        if "xl/workbook.xml" in names:
            wb = ET.fromstring(zf.read("xl/workbook.xml"))
            first_sheet = wb.find(f"{SSML}sheets/{SSML}sheet")
            if first_sheet is not None:
                rid = first_sheet.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                )
                if rid and "xl/_rels/workbook.xml.rels" in names:
                    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
                    for rel in rels:
                        if rel.get("Id") == rid:
                            target = rel.get("Target", "").lstrip("/")
                            if not target.startswith("xl/"):
                                target = "xl/" + target
                            sheet_target = target
                            break

        sheet = ET.fromstring(zf.read(sheet_target))
        rows = []
        for row in sheet.iter(f"{SSML}row"):
            cells = {}
            for cell in row.findall(f"{SSML}c"):
                ref = cell.get("r", "")
                letters = re.match(r"[A-Z]+", ref)
                if not letters:
                    continue
                col = col_letter_to_index(letters.group(0))
                cells[col] = cell
            if not cells:
                continue
            width = max(cells) + 1
            values = []
            for col in range(width):
                cell = cells.get(col)
                if cell is None:
                    values.append("")
                    continue
                ctype = cell.get("t")
                if ctype == "inlineStr":
                    node = cell.find(f"{SSML}is")
                    value = (
                        "".join(t.text or "" for t in node.iter(f"{SSML}t"))
                        if node is not None
                        else ""
                    )
                else:
                    v = cell.find(f"{SSML}v")
                    raw = v.text if v is not None else ""
                    if ctype == "s" and raw:
                        value = shared[int(raw)]
                    else:
                        value = raw or ""
                values.append(value)
            rows.append(values)

    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    return [dict(zip(header, row)) for row in rows[1:]]


def read_csv_file(path):
    """Read a CSV file (handles the UTF-8 BOM written by the sampler)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def read_rows(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"ERROR: file not found: {path}")
    if path.suffix.lower() == ".xlsx":
        return read_xlsx(path)
    if path.suffix.lower() in (".csv", ".tsv"):
        return read_csv_file(path)
    raise SystemExit(f"ERROR: unsupported file type: {path}")


# ---------------------------------------------------------------------------
# Pairing and kappa
# ---------------------------------------------------------------------------

def find_annotation_column(rows, explicit=None):
    """Pick the annotation column from the file's header."""
    if not rows:
        raise SystemExit("ERROR: file has no data rows")
    columns = list(rows[0])
    if explicit:
        for col in columns:
            if col.strip().lower() == explicit.strip().lower():
                return col
        raise SystemExit(
            f"ERROR: column {explicit!r} not found. Available: {columns}"
        )
    for candidate in ANNOTATION_COLUMN_CANDIDATES:
        for col in columns:
            if col.strip().lower() == candidate.lower():
                return col
    raise SystemExit(
        "ERROR: no annotation column found (looked for "
        f"{ANNOTATION_COLUMN_CANDIDATES}). Available: {columns}. "
        "Use --column to specify one."
    )


def row_key(row):
    """Content key used to pair rows across files."""
    return tuple(str(row.get(col, "")).strip() for col in KEY_COLUMNS)


def normalize_label(value):
    return str(value).strip().upper()


def pair_rows(rows_a, rows_b):
    """Pair rows from the two files by content key (run_id + query).

    Within a duplicate group, rows are matched by exact answer text first
    (so duplicated samples with different answers stay distinct), then by
    file order. Returns (pairs, unmatched_a, unmatched_b, duplicate_keys).
    """
    def index(rows):
        grouped = {}
        duplicates = set()
        for row in rows:
            key = row_key(row)
            if key in grouped:
                duplicates.add(key)
            grouped.setdefault(key, []).append(row)
        return grouped, duplicates

    grouped_a, dup_a = index(rows_a)
    grouped_b, dup_b = index(rows_b)
    duplicates = dup_a | dup_b

    pairs, unmatched_a, unmatched_b = [], [], []
    for key in grouped_a:
        rows = grouped_a[key]
        others = grouped_b.get(key, [])
        used = [False] * len(others)
        # Pass 1: match by exact answer text.
        for row_a in rows:
            answer = str(row_a.get("answer", "")).strip()
            for j, row_b in enumerate(others):
                if not used[j] and str(row_b.get("answer", "")).strip() == answer:
                    pairs.append((key, row_a, row_b))
                    used[j] = True
                    break
        # Pass 2: match remaining rows in file order.
        remaining_a = [r for r in rows if not any(p[1] is r for p in pairs)]
        for row_a in remaining_a:
            match = next((j for j, r in enumerate(others) if not used[j]), None)
            if match is None:
                unmatched_a.append(row_a)
            else:
                pairs.append((key, row_a, others[match]))
                used[match] = True
    for key in grouped_b:
        n_a = len(grouped_a.get(key, []))
        unmatched_b.extend(grouped_b[key][n_a:])
    return pairs, unmatched_a, unmatched_b, duplicates


def cohens_kappa(labels_a, labels_b):
    """Cohen's kappa plus observed/expected agreement and the confusion matrix."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must have equal length")
    n = len(labels_a)
    if n == 0:
        return None, None, None, {}

    categories = sorted(set(labels_a) | set(labels_b))
    confusion = {(ca, cb): 0 for ca in categories for cb in categories}
    for la, lb in zip(labels_a, labels_b):
        confusion[(la, lb)] += 1

    p_o = sum(1 for la, lb in zip(labels_a, labels_b) if la == lb) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    p_e = sum(
        (counts_a[c] / n) * (counts_b[c] / n) for c in set(counts_a) & set(counts_b)
    )

    if p_e == 1:
        kappa = 1.0 if p_o == 1 else float("nan")  # degenerate: single category
    else:
        kappa = (p_o - p_e) / (1 - p_e)
    return kappa, p_o, p_e, confusion


def bootstrap_kappa_ci(labels_a, labels_b, n_boot, seed=0):
    """Percentile bootstrap confidence interval for Cohen's kappa."""
    rng = random.Random(seed)
    n = len(labels_a)
    samples = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        kappa, _, _, _ = cohens_kappa(
            [labels_a[i] for i in idx], [labels_b[i] for i in idx]
        )
        if kappa is not None and kappa == kappa:  # skip NaN
            samples.append(kappa)
    if not samples:
        return None
    samples.sort()

    def percentile(p):
        k = (len(samples) - 1) * p
        lo = int(k)
        hi = min(lo + 1, len(samples) - 1)
        frac = k - lo
        return samples[lo] * (1 - frac) + samples[hi] * frac

    return percentile(0.025), percentile(0.975)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render_confusion(confusion, labels_a, labels_b):
    categories = sorted(set(labels_a) | set(labels_b))
    width = max([len(c) for c in categories] + [7])
    lines = ["Confusion matrix (rows = file 1, columns = file 2):"]
    header = " " * (width + 2) + "".join(f"{c:>{width + 2}}" for c in categories)
    lines.append(header)
    for ca in categories:
        cells = "".join(f"{confusion[(ca, cb)]:>{width + 2}}" for cb in categories)
        lines.append(f"{ca:>{width + 2}}  {cells}")
    return "\n".join(lines)


def write_disagreements(pairs, col_a, col_b, path):
    fieldnames = ["run_id", "query", f"{col_a} (file 1)", f"{col_b} (file 2)", "answer"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for key, row_a, row_b in pairs:
            la = normalize_label(row_a.get(col_a, ""))
            lb = normalize_label(row_b.get(col_b, ""))
            if la != lb:
                writer.writerow(
                    {
                        "run_id": row_a.get("run_id", ""),
                        "query": row_a.get("query", ""),
                        f"{col_a} (file 1)": row_a.get(col_a, ""),
                        f"{col_b} (file 2)": row_b.get(col_b, ""),
                        "answer": row_a.get("answer", ""),
                    }
                )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Cohen's kappa between the annotation columns of two files."
    )
    parser.add_argument("file1", help="first annotation file (.xlsx or .csv)")
    parser.add_argument("file2", help="second annotation file (.xlsx or .csv)")
    parser.add_argument(
        "--column",
        help="annotation column name (default: 'Human annotation', then 'annotation')",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="number of bootstrap resamples for a 95%% CI of kappa (default: off)",
    )
    parser.add_argument(
        "--disagreements-out",
        metavar="PATH",
        help="write disagreeing rows to this CSV file",
    )
    args = parser.parse_args(argv)

    rows_a = read_rows(args.file1)
    rows_b = read_rows(args.file2)
    col_a = find_annotation_column(rows_a, args.column)
    col_b = find_annotation_column(rows_b, args.column)
    print(f"File 1: {args.file1}  ({len(rows_a)} rows, column {col_a!r})")
    print(f"File 2: {args.file2}  ({len(rows_b)} rows, column {col_b!r})")

    pairs, unmatched_a, unmatched_b, duplicates = pair_rows(rows_a, rows_b)
    if unmatched_a or unmatched_b:
        print(
            f"WARNING: {len(unmatched_a)} row(s) in file 1 and "
            f"{len(unmatched_b)} row(s) in file 2 could not be paired "
            "and were excluded.",
            file=sys.stderr,
        )

    # Keep only pairs where both annotators gave a non-blank label.
    scored = [
        (key, normalize_label(row_a.get(col_a, "")), normalize_label(row_b.get(col_b, "")))
        for key, row_a, row_b in pairs
    ]
    blank = [(k, la, lb) for k, la, lb in scored if not la or not lb]
    usable = [(k, la, lb) for k, la, lb in scored if la and lb]

    print(f"\nPaired rows: {len(pairs)}")
    if blank:
        print(f"Excluded (blank annotation): {len(blank)}")
    print(f"Used for kappa: {len(usable)}")

    labels_a = [la for _, la, _ in usable]
    labels_b = [lb for _, _, lb in usable]
    kappa, p_o, p_e, confusion = cohens_kappa(labels_a, labels_b)

    print("\nLabel counts (file 1 / file 2):")
    for category in sorted(set(labels_a) | set(labels_b)):
        print(
            f"  {category}: {labels_a.count(category)} / {labels_b.count(category)}"
        )

    print(f"\nObserved agreement (p_o): {p_o:.4f}")
    print(f"Expected agreement (p_e): {p_e:.4f}")
    if kappa is None or kappa != kappa:
        print("Cohen's kappa: undefined (degenerate label distribution)")
    else:
        print(f"Cohen's kappa: {kappa:.4f}")
        if args.bootstrap > 0:
            ci = bootstrap_kappa_ci(labels_a, labels_b, args.bootstrap)
            if ci is not None:
                print(
                    f"95% bootstrap CI ({args.bootstrap} resamples): "
                    f"[{ci[0]:.4f}, {ci[1]:.4f}]"
                )

    print()
    print(render_confusion(confusion, labels_a, labels_b))

    n_disagree = sum(1 for la, lb in zip(labels_a, labels_b) if la != lb)
    print(f"\nDisagreements: {n_disagree} of {len(usable)}")
    if args.disagreements_out:
        write_disagreements(pairs, col_a, col_b, args.disagreements_out)
        print(f"Disagreeing rows written to {args.disagreements_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
