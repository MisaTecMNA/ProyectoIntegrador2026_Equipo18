
"""
Fast multi-year sampler:
Collect up to 5,000 SemMedDB triples per year for 2010..2024 (observed in that year),
and write them all into one CSV.

Strategy:
1) Parse CITATIONS .sql.gz once -> build pmid->year map for target years.
2) Stream PREDICATION .csv.gz once -> for each row, find its year via pmid->year,
   build canonical triple key, collect the first time seen per year.
3) Stop early once all years reach 5,000 (or end of file).

NOTE: This yields triples observed in that year (does NOT verify earliest appearance).
"""

import gzip
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

# -------- Paths --------
PREDICATION_PATH = "/home/xingmeng/knowledge/SemMedDB/semmedVER43_2024_R_PREDICATION.csv.gz"
CITATIONS_SQL_PATH = "/home/xingmeng/knowledge/SemMedDB/semmedVER43_2024_R_CITATIONS.sql.gz"
OUT_PATH = Path("/home/tec/code/misael_space/data/triples_10_24_semmed.csv")

# -------- Parameters --------
TARGET_YEARS = list(range(2010, 2025))  # 2010..2024 inclusive
PER_YEAR_TARGET = 70000                   # <- changed from 3000 to 5000
PRED_CHUNKSIZE = 200_000
SQL_BATCH_ROWS = 5_000  # how many parsed rows to emit per batch while scanning SQL

# Optional: filter to specific predicates (e.g., {"PREVENTS","TREATS"}), or None for ALL
PREDICATE_FILTER: Optional[set] = None

# Symmetric predicates (order-independent S,P,O)
SYMMETRIC = {"INTERACTS_WITH", "COEXISTS_WITH", "ASSOCIATED_WITH", "ASSOCIATED_WITH_INFER"}

# PREDICATION CSV expected columns (most dumps ship WITHOUT header)
PREDICATION_COLS = [
    "PREDICATION_ID", "SENTENCE_ID", "PMID", "PREDICATE",
    "SUBJECT_CUI", "SUBJECT_NAME", "SUBJECT_SEMTYPE", "SUBJECT_NOVELTY",
    "OBJECT_CUI", "OBJECT_NAME", "OBJECT_SEMTYPE", "OBJECT_NOVELTY",
    "PREDICATION_NOVELTY", "TYPE", "EVAL_TYPE"
]

YEAR_RE = re.compile(r"(19|20)\d{2}")


# ---------------- utils ----------------
def _to_int_or_none(x) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None

def extract_year(pyear: Optional[int | str], dp: Optional[str], edat: Optional[str]) -> Optional[int]:
    """Try PYEAR, then any 4-digit year in DP, then EDAT."""
    if pyear not in (None, r"\N", ""):
        y = _to_int_or_none(pyear)
        if y:
            return y
    for val in (dp, edat):
        if val and val != r"\N":
            m = YEAR_RE.search(str(val))
            if m:
                return int(m.group(0))
    return None

def canonical_key(subj_cui: str, obj_cui: str, predicate: str) -> Tuple[str, str, str]:
    predicate = predicate.upper()
    if predicate in SYMMETRIC:
        return (subj_cui, predicate, obj_cui) if (subj_cui or "") <= (obj_cui or "") else (obj_cui, predicate, subj_cui)
    return (subj_cui, predicate, obj_cui)


# ------ CITATIONS: build pmid->year (only for target years) ------
def _split_values_tuples(values_blob: str) -> List[str]:
    inner = values_blob.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    return re.split(r"\)\s*,\s*\(", inner)

def _split_fields_preserving_quotes(tuple_str: str) -> List[str]:
    parts = re.findall(r"(?:'[^']*'|[^,]+)", tuple_str)
    return [p.strip().strip("'") for p in parts]

def build_pmid_to_year(sql_gz_path: str, years: set[int]) -> Dict[int, int]:
    """
    Stream the CITATIONS .sql.gz and build a map: PMID -> YEAR
    Only keep PMIDs whose year is in `years`.
    """
    pmid2year: Dict[int, int] = {}
    values_pat = re.compile(r"INSERT\s+INTO\s+`?CITATIONS`?\s+VALUES\s*(.+);", re.IGNORECASE)

    with gzip.open(sql_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        batch_rows = 0
        for line in f:
            if "INSERT" not in line.upper():
                continue
            m = values_pat.search(line)
            if not m:
                continue
            blob = m.group(1)
            tuples = _split_values_tuples(blob)
            for tup in tuples:
                fields = _split_fields_preserving_quotes(tup)
                if len(fields) < 5:
                    continue
                pmid = _to_int_or_none(fields[0])
                if pmid is None:
                    continue
                dp, edat, pyear = fields[2], fields[3], fields[4]
                y = extract_year(pyear, dp, edat)
                if y in years:
                    pmid2year[pmid] = y
                batch_rows += 1
            if batch_rows >= 2_000_000:  # periodic progress print
                print(f"[CITATIONS] parsed ~{batch_rows:,} rows, pmid->year entries: {len(pmid2year):,}")
                batch_rows = 0

    print(f"[CITATIONS] done. pmid->year map size (target years): {len(pmid2year):,}")
    return pmid2year


# --------- PREDICATION streaming ---------
def stream_predication(pred_csv_gz_path: str, chunksize: int = PRED_CHUNKSIZE) -> Iterator[pd.DataFrame]:
    with gzip.open(pred_csv_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        it = pd.read_csv(f, chunksize=chunksize, names=PREDICATION_COLS, header=None)
        for ch in it:
            yield ch


# ---------------- main ----------------
def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    target_years_set = set(TARGET_YEARS)

    # 1) Build PMID -> year map for target years
    print(f"→ Building PMID→year map for years {TARGET_YEARS} from: {CITATIONS_SQL_PATH}")
    pmid2year = build_pmid_to_year(CITATIONS_SQL_PATH, target_years_set)
    if not pmid2year:
        print("No PMIDs found for target years. Exiting.")
        return

    # Per-year structures
    per_year_triples: Dict[int, Dict[Tuple[str, str, str], dict]] = {y: {} for y in TARGET_YEARS}
    per_year_counts: Dict[int, int] = {y: 0 for y in TARGET_YEARS}

    # Helper to check if all years reached target
    def all_done() -> bool:
        return all(per_year_counts[y] >= PER_YEAR_TARGET for y in TARGET_YEARS)

    # 2) Stream PREDICATION once and collect per-year
    print(f"→ Streaming PREDICATION from: {PREDICATION_PATH}")
    chunks = 0
    for ch in stream_predication(PREDICATION_PATH):
        chunks += 1

        # Cast PMID once and map to year
        ch["PMID"] = ch["PMID"].apply(_to_int_or_none)
        ch["YEAR"] = ch["PMID"].map(pmid2year)

        # Keep only rows whose YEAR is one of our target years and that year still needs rows
        ch = ch[ch["YEAR"].isin(target_years_set)]
        if ch.empty:
            if chunks % 25 == 0:
                print(f"[PREDICATION] chunk {chunks:,}: no rows for target years")
            continue

        # Optional predicate filter
        if PREDICATE_FILTER:
            ch["PREDICATE"] = ch["PREDICATE"].astype(str).str.upper().str.strip()
            ch = ch[ch["PREDICATE"].isin(PREDICATE_FILTER)]
            if ch.empty:
                continue

        # Iterate rows
        for row in ch.itertuples(index=False):
            year = int(row.YEAR)
            # Skip if this year already complete
            if per_year_counts[year] >= PER_YEAR_TARGET:
                continue

            subj_cui = "" if row.SUBJECT_CUI is None else str(row.SUBJECT_CUI)
            obj_cui  = "" if row.OBJECT_CUI is None else str(row.OBJECT_CUI)
            predicate = str(row.PREDICATE).upper().strip()

            key = canonical_key(subj_cui, obj_cui, predicate)
            year_dict = per_year_triples[year]

            if key in year_dict:
                continue

            this_pmid = _to_int_or_none(row.PMID)

            # Store triple record (now including citation_pmid)
            year_dict[key] = {
                "subject_cui": subj_cui,
                "subject_name": row.SUBJECT_NAME,
                "subject_semtype": row.SUBJECT_SEMTYPE,
                "predicate": predicate,
                "object_cui": obj_cui,
                "object_name": row.OBJECT_NAME,
                "object_semtype": row.OBJECT_SEMTYPE,
                "first_year": year,                 # observed in this year
                "first_pmid": this_pmid,            # legacy name kept
                "citation_pmid": this_pmid,         # <-- added explicitly from CITATIONS domain (same PMID)
                "first_issn": None,                 # keep None for speed; can be backfilled later if needed
                "first_sentence_id": row.SENTENCE_ID,
            }
            per_year_counts[year] += 1

            # Early stop if all years complete
            if all_done():
                break

        if chunks % 25 == 0:
            progress_preview = ", ".join(f"{y}:{per_year_counts[y]}" for y in TARGET_YEARS[:5])
            print(f"[PREDICATION] chunk {chunks:,} progress → {progress_preview} ...")

        if all_done():
            print(f"[PREDICATION] early stop at chunk {chunks:,} (all years reached {PER_YEAR_TARGET}).")
            break

    # 3) Combine and write CSV
    rows = []
    for y in TARGET_YEARS:
        rows.extend(per_year_triples[y].values())

    if not rows:
        print("No triples collected for target years.")
        return

    df = pd.DataFrame.from_records(rows)
    # Optional ordering for reproducibility
    df.sort_values(["first_year", "predicate", "subject_cui", "object_cui", "first_pmid"],
                   inplace=True, kind="mergesort")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} triples "
          f"(up to {PER_YEAR_TARGET} per year for 2010–2024) to: {OUT_PATH}")
    print("Per-year counts:", {y: per_year_counts[y] for y in TARGET_YEARS})


if __name__ == "__main__":
    main()
