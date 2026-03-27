import os
import json
import time
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import Compare_abstracts
import ollama
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

# --- Constants ---
PARAMS = {
    "api_key": "tTekBojph8alEiFnymyAwn",
    "mailto": "benjamin.collins3@mail.mcgill.ca"
}
SESSION = requests.Session()  # Reuse TCP connections across all requests
SESSION.params.update(PARAMS)


# --- API Helpers ---

def _get_json(url, **kwargs):
    """GET a URL and return parsed JSON, or None on failure."""
    try:
        r = SESSION.get(url, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def get_author_data(author_id, cached_h_index=None):
    """
    Fetch an author's h-index and abstracts.

    If cached_h_index is supplied (e.g. read from mcgill_authors.csv) the
    author-level API call is skipped entirely — only the works pages are fetched.

    Returns:
        (h_index, abstracts) tuple, or (None, None) on failure.
    """
    if cached_h_index is not None:
        # h-index already known from CSV — skip the author endpoint call
        h_index = cached_h_index
    else:
        author_data = _get_json(f"https://api.openalex.org/authors/{author_id}")
        if not author_data:
            return None, None
        h_index = author_data.get("summary_stats", {}).get("h_index")

    # Paginate works — request max page size to minimise round-trips
    works = []
    url = f"https://api.openalex.org/works?filter=author.id:{author_id}&per-page=200&select=abstract_inverted_index"

    while url:
        data = _get_json(url)
        if not data:
            break
        works.extend(data.get("results", []))
        url = data.get("meta", {}).get("next")

    abstracts = []
    for work in works:
        idx = work.get("abstract_inverted_index")
        if idx:
            text = _invert_abstract_index(idx)
            if text:
                abstracts.append({"abstract": text})

    return h_index, abstracts


def _invert_abstract_index(abstract_index):
    """Convert OpenAlex abstract_inverted_index to plain text."""
    if not abstract_index:
        return None
    positions = {}
    for word, pos_list in abstract_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))


# --- McGill Author Discovery ---

# OpenAlex institution ID for McGill University
MCGILL_INSTITUTION_ID = "I5023651"


def _fetch_mcgill_author_page(cursor):
    """
    Fetch one page of McGill authors from OpenAlex using cursor-based pagination.
    Returns (results, next_cursor) or ([], None) on failure.
    """
    url = (
        f"https://api.openalex.org/authors"
        f"?filter=last_known_institutions.id:{MCGILL_INSTITUTION_ID}"
        f"&per-page=200"
        f"&select=id,display_name,summary_stats,works_count,cited_by_count,topics"
        f"&cursor={cursor}"
    )
    data = _get_json(url)
    if not data:
        return [], None
    results = data.get("results", [])
    next_cursor = data.get("meta", {}).get("next_cursor")
    return results, next_cursor


def _parse_author_record(author):
    """
    Extract the fields we care about from a raw OpenAlex author object.

    Fields of study come from the author's top-weighted topics. OpenAlex groups
    topics into domains/fields; we take the top 3 distinct field names so the
    CSV stays readable.
    """
    stats = author.get("summary_stats", {})

    # Collect fields of study from topics (each topic carries a 'field' sub-object)
    seen_fields = []
    for topic_entry in author.get("topics", []):
        field_name = (
            topic_entry.get("field", {}).get("display_name")
            or topic_entry.get("subfield", {}).get("display_name")
        )
        if field_name and field_name not in seen_fields:
            seen_fields.append(field_name)
        if len(seen_fields) == 3:
            break
    fields_of_study = "; ".join(seen_fields) if seen_fields else ""

    # Strip the URL prefix so we get a bare ID like 'A1234567890'
    raw_id = author.get("id", "")
    author_id = raw_id.replace("https://openalex.org/", "")

    return {
        "author_id":           author_id,
        "name":                author.get("display_name", ""),
        "fields_of_study":     fields_of_study,
        "publication_count":   author.get("works_count", 0),
        "h_index":             stats.get("h_index", 0),
        "citation_count":      author.get("cited_by_count", 0),
        # Populated later by the similarity pipeline; None until then
        "avg_self_similarity": None,
        "std_self_similarity": None,
    }


def _load_csv_records(output_csv):
    """
    Read an existing mcgill_authors CSV and return a list of record dicts.
    Numeric fields are cast back to their correct types so downstream code
    stays consistent.  The similarity columns are optional — they are absent
    on the first run and populated after similarities are computed.
    """
    records = []
    with open(output_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["publication_count"] = int(row["publication_count"])
            row["h_index"]           = int(row["h_index"])
            row["citation_count"]    = int(row["citation_count"])
            # Cast similarity columns only when they are already present
            for col in ("avg_self_similarity", "std_self_similarity"):
                if col in row:
                    row[col] = float(row[col]) if row[col] not in ("", "None") else None
            records.append(row)
    return records


def get_mcgill_authors(output_csv="mcgill_authors.csv", max_workers=8, refresh=False):
    """
    Return McGill author records, reading from the CSV cache when available.

    On the first run (or when refresh=True) the full OpenAlex crawl is executed
    and results are written to output_csv.  On every subsequent run the CSV is
    read directly — no API calls are made for the author list.

    Cursor-based pagination is used instead of offset paging — it is faster and
    avoids the 10,000-result offset cap that OpenAlex enforces.

    Args:
        output_csv (str):  Path for the CSV cache file.
        max_workers (int): Thread-pool size used to parse records in parallel.
        refresh (bool):    Force a fresh API crawl even if the CSV exists.

    Returns:
        list[dict]: Author records with keys author_id, name, fields_of_study,
                    publication_count, h_index, citation_count,
                    avg_self_similarity, std_self_similarity.
                    The last two are None until the similarity pipeline runs.
    """
    # ── Cache hit: return CSV contents without touching the API ──────────────
    if not refresh and os.path.exists(output_csv):
        records = _load_csv_records(output_csv)
        print(f"Loaded {len(records):,} McGill authors from cache '{output_csv}' (pass refresh=True to re-fetch)")
        return records

    # ── Cache miss: crawl OpenAlex ────────────────────────────────────────────
    print("Fetching McGill author list from OpenAlex ...")

    # Phase 1: walk cursor pages (must be sequential — each cursor comes from
    # the previous response)
    all_authors_raw = []
    cursor = "*"  # OpenAlex uses "*" as the initial cursor token

    while cursor:
        results, cursor = _fetch_mcgill_author_page(cursor)
        if not results:
            break
        all_authors_raw.extend(results)
        print(f"  ... retrieved {len(all_authors_raw):,} authors so far", end="\r")

    print(f"\nTotal authors fetched: {len(all_authors_raw):,}")

    if not all_authors_raw:
        print("No authors found — check the institution ID or network access.")
        return []

    # Phase 2: parse records in parallel (lightweight CPU work)
    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_parse_author_record, a) for a in all_authors_raw]
        for future in as_completed(futures):
            rec = future.result()
            if rec:
                records.append(rec)

    # Sort by h-index descending so the CSV is immediately useful
    records.sort(key=lambda r: r["h_index"], reverse=True)

    # Phase 3: write CSV
    fieldnames = [
        "author_id", "name", "fields_of_study",
        "publication_count", "h_index", "citation_count",
        "avg_self_similarity", "std_self_similarity",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {len(records):,} author records to '{output_csv}'")
    return records


# --- Embedding & Similarity ---

def _batch_get_vectors(abstracts, embedding_cache):
    """
    Vectorise all abstracts, hitting the cache first and only calling the
    embedding model for cache misses — instead of calling get_vector() one
    at a time inside a loop.
    """
    texts = [item["abstract"] for item in abstracts]

    # Identify which texts are cache misses
    misses = [t for t in texts if t not in embedding_cache]

    if misses:
        # Batch-embed all misses in one model call where the model supports it;
        # fall back to individual calls if the model doesn't expose batch input.
        try:
            response = ollama.embed(model="embeddinggemma", input=misses)
            for text, vec in zip(misses, response.embeddings):
                embedding_cache[text] = vec
        except Exception:
            # Graceful degradation: embed one-by-one
            for text in misses:
                try:
                    response = ollama.embed(model="embeddinggemma", input=text)
                    embedding_cache[text] = response.embeddings[0]
                except Exception:
                    embedding_cache[text] = None

    vectors = [embedding_cache.get(t) for t in texts]
    return [v for v in vectors if v is not None]


# Directory where per-author similarity caches are stored
SIMILARITY_DIR = "Author_self_similarity"


def _similarity_npz_path(author_name, author_id):
    """Return the .npz cache path for a given author."""
    safe_name = author_name.replace(" ", "_").replace("/", "-")
    filename = f"{safe_name}_{author_id}.npz"
    return os.path.join(SIMILARITY_DIR, filename)


def _similarity_csv_path(author_name, author_id):
    """Return the .csv output path for a given author."""
    safe_name = author_name.replace(" ", "_").replace("/", "-")
    filename = f"{safe_name}_{author_id}.csv"
    return os.path.join(SIMILARITY_DIR, filename)


def calculate_similarity_matrix(abstracts_list, embedding_cache,
                                 author_name=None, author_id=None):
    """
    Build an upper-triangle cosine similarity matrix.

    Persistence strategy — two files per author inside SIMILARITY_DIR:
      • <Name>_<id>.npz  — compact binary cache (float32 upper-triangle values
                            + their (i, j) index pairs).  Used to skip
                            recomputation on subsequent runs.  ~10× smaller
                            than the equivalent CSV.
      • <Name>_<id>.csv  — human-readable export with columns i, j, similarity.
                            Written once (when the npz is first created).

    Args:
        abstracts_list (list[dict]): Abstracts for one author.
        embedding_cache (dict):      Shared embedding cache.
        author_name (str|None):      Used to build the output filenames.
        author_id   (str|None):      Used to build the output filenames.

    Returns:
        pd.DataFrame | None: Full N×N similarity DataFrame (upper-triangle
                             populated, lower triangle and diagonal are zero),
                             or None if there are fewer than 2 valid abstracts.
    """
    os.makedirs(SIMILARITY_DIR, exist_ok=True)

    save_files = author_name and author_id
    npz_path   = _similarity_npz_path(author_name, author_id) if save_files else None
    csv_path   = _similarity_csv_path(author_name, author_id) if save_files else None

    # ── Cache hit: reconstruct DataFrame from .npz without re-embedding ───────
    if save_files and os.path.exists(npz_path):
        data = np.load(npz_path)
        rows, cols, vals = data["rows"], data["cols"], data["vals"]
        n = int(data["n"])
        sim = np.zeros((n, n), dtype=np.float64)
        sim[rows, cols] = vals
        return pd.DataFrame(sim)

    # ── Cache miss: compute from scratch ─────────────────────────────────────
    valid = [a for a in abstracts_list if a.get("abstract", "").strip()]
    if len(valid) < 2:
        return None

    vectors = _batch_get_vectors(valid, embedding_cache)
    if len(vectors) < 2:
        return None

    n = len(vectors)
    mat    = np.array(vectors, dtype=np.float32)
    norms  = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normed = mat / norms
    sim_f32 = normed @ normed.T          # float32 matmul — fast and compact
    sim_f32  = np.triu(sim_f32, k=1)

    # Extract upper-triangle indices and values for storage
    rows, cols = np.triu_indices(n, k=1)
    vals = sim_f32[rows, cols]           # float32, ~4 bytes each

    # ── Persist .npz (binary, compressed, ~10× smaller than CSV) ─────────────
    if save_files:
        np.savez_compressed(npz_path, rows=rows, cols=cols, vals=vals, n=np.int32(n))

        # ── Write human-readable CSV (written once alongside the .npz) ───────
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["abstract_i", "abstract_j", "cosine_similarity"])
            for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
                writer.writerow([r, c, round(float(v), 6)])

    sim_f64 = sim_f32.astype(np.float64)
    return pd.DataFrame(sim_f64)


# --- Main Pipeline ---

def _process_author(author_id, cached_h_index=None, author_name=None):
    """Worker function for a single author — runs inside a thread."""
    h_index, abstracts = get_author_data(author_id, cached_h_index=cached_h_index)
    if h_index is None:
        print(f"Skipping {author_id}: could not retrieve author data")
        return None

    if not abstracts:
        print(f"Skipping {author_id}: no abstracts found")
        return None

    return author_id, author_name or author_id, h_index, abstracts


def _write_mcgill_csv(records, output_csv):
    """Rewrite the full mcgill_authors CSV, including any similarity columns."""
    fieldnames = [
        "author_id", "name", "fields_of_study",
        "publication_count", "h_index", "citation_count",
        "avg_self_similarity", "std_self_similarity",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def plot_similarities_vs_h_index(author_ids, h_index_cache=None,
                                 name_cache=None, mcgill_records=None,
                                 output_csv="mcgill_authors.csv",
                                 max_workers=8):
    """
    Compute pairwise cosine similarities, plot them vs h-index, and write
    avg/std self-similarity back to mcgill_authors.csv.

    Authors whose avg_self_similarity is already filled in mcgill_records are
    skipped — no embeddings or API calls are made for them.

    Args:
        author_ids (list[str]):           Author IDs to process.
        h_index_cache (dict, optional):   author_id -> h_index.
        name_cache (dict, optional):      author_id -> display name.
        mcgill_records (list[dict]|None): Live records list; stats are written
                                          back in-place and the CSV is rewritten
                                          after all authors are processed.
        output_csv (str):                 Path to mcgill_authors.csv.
        max_workers (int):                Thread-pool size for fetching.
    """
    h_index_cache = h_index_cache or {}
    name_cache    = name_cache    or {}

    # Build a fast lookup from mcgill_records so we can write stats back
    # and skip authors that already have similarity data
    record_by_id = {r["author_id"]: r for r in (mcgill_records or [])}

    # Determine which authors still need similarity computation
    ids_to_process = []
    for aid in author_ids:
        rec = record_by_id.get(aid)
        if rec and rec.get("avg_self_similarity") not in (None, ""):
            # Already computed in a previous run — skip entirely
            continue
        ids_to_process.append(aid)

    skipped = len(author_ids) - len(ids_to_process)
    if skipped:
        print(f"Skipping {skipped:,} authors with similarity stats already in CSV")

    # Load embedding cache once for the whole run
    embedding_cache = Compare_abstracts.load_embedding_cache(cache_path="embedding_cache.json")

    # Fetch works for authors that still need processing (concurrent)
    author_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_author, aid,
                h_index_cache.get(aid),
                name_cache.get(aid)
            ): aid
            for aid in ids_to_process
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                author_results.append(result)

    # Compute similarities and accumulate plot data
    all_h, all_sims, avg_h, avg_sims = [], [], [], []

    for author_id, author_name, h_index, abstracts in author_results:
        sim_df = calculate_similarity_matrix(
            abstracts, embedding_cache,
            author_name=author_name, author_id=author_id
        )
        if sim_df is None or sim_df.empty:
            print(f"Skipping {author_id}: similarity matrix empty")
            continue

        n = len(sim_df)
        rows, cols = np.triu_indices(n, k=1)
        sims = sim_df.values[rows, cols]        # numpy array, not list

        avg_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))

        # Write stats back into the live records dict
        if author_id in record_by_id:
            record_by_id[author_id]["avg_self_similarity"] = round(avg_sim, 6)
            record_by_id[author_id]["std_self_similarity"] = round(std_sim, 6)

        all_h.extend([h_index] * len(sims))
        all_sims.extend(sims.tolist())
        avg_h.append(h_index)
        avg_sims.append(avg_sim)

    # Persist embeddings and rewrite the CSV with updated similarity columns
    Compare_abstracts.save_embedding_cache(embedding_cache)

    if mcgill_records and author_results:
        _write_mcgill_csv(mcgill_records, output_csv)
        print(f"Updated similarity stats written to '{output_csv}'")

    if not all_sims:
        print("No new similarity data to plot.")
        return

    # Plot
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    #ax1.scatter(all_h, all_sims, alpha=0.7)
    #ax1.set_xlabel("H-Index")
    #ax1.set_ylabel("Cosine Similarity")
    #ax1.set_title("Individual Work Similarities vs H-Index")
    #ax1.grid(True, alpha=0.3)

    #ax2.scatter(avg_h, avg_sims, color="red", alpha=0.7)
    #ax2.set_xlabel("H-Index")
    #ax2.set_ylabel("Average Cosine Similarity")
    #ax2.set_title("Average Similarity vs H-Index")
    #ax2.grid(True, alpha=0.3)

    #plt.tight_layout()
    #plt.savefig("results.png")
    #plt.show()


# --- Entry point ---

# Fetch all McGill authors (reads from mcgill_authors.csv if it exists).
mcgill_records = get_mcgill_authors(output_csv="mcgill_authors_stats.csv")


if mcgill_records:
    LIST_OF_AUTHOR_IDS = [r["author_id"] for r in mcgill_records]
    # Build lookups so the similarity pipeline can skip the per-author API call
    # and use the author's real name in output filenames
    H_INDEX_CACHE = {r["author_id"]: r["h_index"] for r in mcgill_records}
    NAME_CACHE    = {r["author_id"]: r["name"]    for r in mcgill_records}
else:
    print("Falling back to hard-coded author list.")
    LIST_OF_AUTHOR_IDS = [
        "a5048442336", "a5101834127", "a5101839428", "a5039389811", "a5103029167",
        "a5070615546", "a5111998194", "a5090620545", "a5040277346", "a5070728216",
        "a5102956975", "a5040143352", "a5090620545", "a5063905190", "a5031796630",
        "a5018142180", "a5114002649", "a5062597899", "a5089292437", "a5036537125",
        "a5071633297", "a5037335973",
    ]
    H_INDEX_CACHE = {}
    NAME_CACHE    = {}


plot_similarities_vs_h_index(
    LIST_OF_AUTHOR_IDS,
    h_index_cache=H_INDEX_CACHE,
    name_cache=NAME_CACHE,
    mcgill_records=mcgill_records,
    output_csv="mcgill_authors_stats.csv",
)