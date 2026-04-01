import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import json

# =============================================================================
# CONFIGURATION
# All user-facing settings live here. Edit these values to customise behaviour
# without touching any function code below.
# =============================================================================

# --- OpenAlex API credentials ------------------------------------------------
# api_key: your OpenAlex Polite Pool key. Requests without a key still work but
#   share a lower-priority pool. Get a free key at https://openalex.org/
# mailto:  your email address. OpenAlex uses this to contact you if your
#   requests cause problems; it also grants access to the Polite Pool.
OPENALEX_API_KEY = "tTekBojph8alEiFnymyAwn"
OPENALEX_MAILTO  = "benjamin.collins3@mail.mcgill.ca"

# --- Institution filter ------------------------------------------------------
# The OpenAlex institution ID used to filter authors.
# McGill University: "I5023651"
# Find other institution IDs at https://openalex.org/institutions
INSTITUTION_ID = "I5023651"

# --- Output files & folders --------------------------------------------------
# AUTHORS_CSV: path to the main CSV that stores one row per author with their
#   bibliometric stats and self-similarity scores. Created on first run; read
#   from cache on subsequent runs; updated in-place as similarities are computed.
AUTHORS_CSV = "mcgill_authors_SentenceTransformers.csv"

# SIMILARITY_DIR: folder where per-author similarity data is stored.
#   Each author gets two files inside this folder:
#     {Name}_{ID}.npz — compressed binary cache used to skip recomputation.
#     {Name}_{ID}.csv — human-readable pairwise similarity table.
SIMILARITY_DIR = "Author_self_similarity_SentenceTransformers"

# EMBEDDING_CACHE_PATH: JSON file that caches abstract->embedding-vector
#   mappings so abstracts already embedded in a previous run are not re-sent to
#   the model. Can grow large over time; delete to force full re-embedding.
EMBEDDING_CACHE_PATH = "embedding_cache_SentenceTransformers.json"

# RESULTS_PLOT_PATH: file path where the similarity-vs-h-index scatter plot is
#   saved. Supports any matplotlib-supported extension (.png, .pdf, .svg).
RESULTS_PLOT_PATH = "results_SentenceTransformers.png"

# --- Embedding model ---------------------------------------------------------
# SENTENCE_TRANSFORMER_MODEL: any model name accepted by the sentence-transformers
#   library (https://www.sbert.net/docs/pretrained_models.html).
#   "all-mpnet-base-v2" -- 768-dim, best quality in the all-* family.
#   "all-MiniLM-L6-v2"  -- 384-dim, ~5x faster, slightly lower quality.
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"

# EMBEDDING_BATCH_SIZE: number of abstracts passed to the model in one forward
#   pass. Increase for faster throughput on a GPU; decrease if you run out of
#   memory. Has no effect on output quality.
EMBEDDING_BATCH_SIZE = 64

# --- Similarity computation --------------------------------------------------
# MIN_ABSTRACTS: minimum number of unique abstracts an author must have to be
#   included in the similarity analysis. Authors below this threshold are still
#   recorded in the CSV (with their abstract_count filled in) but their
#   avg_self_similarity and std_self_similarity will be left as None.
#   Must be >= 2 (a pairwise matrix needs at least two items).
MIN_ABSTRACTS = 3

# SIMILARITY_DECIMAL_PLACES: number of decimal places used when writing
#   avg_self_similarity and std_self_similarity values to the CSV.
SIMILARITY_DECIMAL_PLACES = 6

# --- Parallelism -------------------------------------------------------------
# MAX_WORKERS: size of the thread pool used for concurrent HTTP requests.
#   Increasing this speeds up the initial author/works fetch but may trigger
#   rate-limiting from the OpenAlex API. 8 is a safe default.
MAX_WORKERS = 8

# --- OpenAlex pagination -----------------------------------------------------
# WORKS_PAGE_SIZE: number of works returned per API page when fetching an
#   author's works. Maximum allowed by OpenAlex is 200.
WORKS_PAGE_SIZE = 200

# AUTHORS_PAGE_SIZE: number of authors returned per API page when crawling the
#   institution's author list. Maximum allowed by OpenAlex is 200.
AUTHORS_PAGE_SIZE = 200

# MAX_FIELDS_OF_STUDY: how many distinct field-of-study labels to record per
#   author. OpenAlex returns topics sorted by relevance weight; we take the top
#   N distinct field names. Increase for more detail; decrease for a tidier CSV.
MAX_FIELDS_OF_STUDY = 4

# =============================================================================
# END OF CONFIGURATION -- do not edit below unless changing functionality
# =============================================================================

# Derived constants (built from config; not intended for direct editing)
_OPENALEX_PARAMS = {"api_key": OPENALEX_API_KEY, "mailto": OPENALEX_MAILTO}

SESSION = requests.Session()
SESSION.params.update(_OPENALEX_PARAMS)

_EMBED_MODEL = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

# -----------------------------------------------------------------------------
# Embedding cache I/O -- owned here, not delegated to the old module, because:
#   a) that module hashes keys with SHA-256; we store raw text as the key.
#      Mixing both formats in one JSON file silently breaks every cache lookup.
#   b) that module embeds via ollama, not sentence-transformers, producing
#      vectors in a different space that cannot be compared against ours.
# -----------------------------------------------------------------------------
 
def _load_embedding_cache():
    """Load the embedding cache from EMBEDDING_CACHE_PATH, or return {} on miss."""
    if not os.path.exists(EMBEDDING_CACHE_PATH):
        return {}
    try:
        with open(EMBEDDING_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
 
 
def _save_embedding_cache(cache):
    """Persist the embedding cache to EMBEDDING_CACHE_PATH."""
    with open(EMBEDDING_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)
 
# Single source of truth for all column names in AUTHORS_CSV.
# Every function that reads or writes that file uses this list, so adding a
# column here is the only change needed to propagate it everywhere.
_AUTHORS_CSV_FIELDS = [
    "author_id",
    "name",
    "fields_of_study",
    "publication_count",
    "h_index",
    "citation_count",
    "abstract_count",
    "avg_self_similarity",
    "std_self_similarity",
]
 
 
# =============================================================================
# API HELPERS
# =============================================================================
 
def _get_json(url, **kwargs):
    """GET a URL via the shared session and return parsed JSON, or None."""
    try:
        r = SESSION.get(url, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None
 
 
def _invert_abstract_index(abstract_index):
    """Reconstruct plain text from an OpenAlex abstract_inverted_index dict."""
    if not abstract_index:
        return None
    positions = {}
    for word, pos_list in abstract_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))
 
 
def get_author_data(author_id, cached_h_index=None):
    """
    Fetch an author's h-index and list of abstracts from OpenAlex.
 
    If cached_h_index is supplied (read from AUTHORS_CSV) the author-level API
    call is skipped entirely -- only the works pages are fetched, saving one
    round-trip per author.
 
    Returns:
        (h_index, abstracts) where abstracts is a list of {'abstract': str}
        dicts. Returns (None, None) on API failure.
    """
    if cached_h_index is not None:
        h_index = cached_h_index
    else:
        data = _get_json(f"https://api.openalex.org/authors/{author_id}")
        if not data:
            return None, None
        h_index = data.get("summary_stats", {}).get("h_index")
 
    works = []
    url = (
        f"https://api.openalex.org/works"
        f"?filter=author.id:{author_id}"
        f"&per-page={WORKS_PAGE_SIZE}"
        f"&select=abstract_inverted_index"
    )
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
 
 
# =============================================================================
# INSTITUTION AUTHOR DISCOVERY
# =============================================================================
 
def _fetch_institution_author_page(cursor):
    """
    Fetch one cursor page of authors affiliated with INSTITUTION_ID.
    Returns (results, next_cursor) or ([], None) on failure.
    """
    url = (
        f"https://api.openalex.org/authors"
        f"?filter=last_known_institutions.id:{INSTITUTION_ID}"
        f"&per-page={AUTHORS_PAGE_SIZE}"
        f"&select=id,display_name,summary_stats,works_count,cited_by_count,topics"
        f"&cursor={cursor}"
    )
    data = _get_json(url)
    if not data:
        return [], None
    return data.get("results", []), data.get("meta", {}).get("next_cursor")
 
 
def _parse_author_record(author):
    """
    Extract CSV fields from a raw OpenAlex author object.
 
    abstract_count and similarity columns are not available at crawl time and
    are initialised to None; they are filled in later by the similarity pipeline.
    """
    stats = author.get("summary_stats", {})
 
    seen_fields = []
    for topic_entry in author.get("topics", []):
        field_name = (
            topic_entry.get("field", {}).get("display_name")
            or topic_entry.get("subfield", {}).get("display_name")
        )
        if field_name and field_name not in seen_fields:
            seen_fields.append(field_name)
        if len(seen_fields) == MAX_FIELDS_OF_STUDY:
            break
 
    raw_id    = author.get("id", "")
    author_id = raw_id.replace("https://openalex.org/", "")
 
    return {
        "author_id":           author_id,
        "name":                author.get("display_name", ""),
        "fields_of_study":     "; ".join(seen_fields) if seen_fields else "",
        "publication_count":   author.get("works_count", 0),
        "h_index":             stats.get("h_index", 0),
        "citation_count":      author.get("cited_by_count", 0),
        "abstract_count":      None,
        "avg_self_similarity": None,
        "std_self_similarity": None,
    }
 
 
def _load_csv_records(csv_path):
    """
    Read an existing authors CSV and return a list of record dicts with correct
    Python types.  Columns added in later versions (abstract_count, similarity
    columns) are handled gracefully if absent from an older CSV file.
    """
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for col in ("publication_count", "h_index", "citation_count"):
                row[col] = int(row[col]) if row.get(col) not in ("", "None", None) else 0
            if "abstract_count" in row:
                raw = row["abstract_count"]
                row["abstract_count"] = int(raw) if raw not in ("", "None", None) else None
            for col in ("avg_self_similarity", "std_self_similarity"):
                if col in row:
                    raw = row[col]
                    row[col] = float(raw) if raw not in ("", "None", None) else None
            records.append(row)
    return records
 
 
def _write_authors_csv(records, csv_path):
    """Write (or overwrite) the authors CSV with all _AUTHORS_CSV_FIELDS columns."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_AUTHORS_CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
 
 
def get_mcgill_authors(refresh=False):
    """
    Return author records for the configured INSTITUTION_ID, using AUTHORS_CSV
    as a persistent cache.
 
    On the first run (or when refresh=True) the full OpenAlex cursor crawl is
    executed and the CSV is written.  On subsequent runs the CSV is read
    directly -- zero API calls are made for the author list.
 
    Args:
        refresh (bool): Force a fresh crawl even if AUTHORS_CSV already exists.
 
    Returns:
        list[dict]: One dict per author containing all _AUTHORS_CSV_FIELDS keys.
    """
    if not refresh and os.path.exists(AUTHORS_CSV):
        records = _load_csv_records(AUTHORS_CSV)
        print(
            f"Loaded {len(records):,} authors from cache '{AUTHORS_CSV}' "
            f"(pass refresh=True to re-fetch)"
        )
        return records
 
    print(f"Fetching author list for institution {INSTITUTION_ID} from OpenAlex ...")
 
    all_raw = []
    cursor  = "*"
    while cursor:
        results, cursor = _fetch_institution_author_page(cursor)
        if not results:
            break
        all_raw.extend(results)
        print(f"  ... {len(all_raw):,} authors retrieved", end="\r")
    print(f"\nTotal authors fetched: {len(all_raw):,}")
 
    if not all_raw:
        print("No authors found -- check INSTITUTION_ID or network access.")
        return []
 
    records = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_parse_author_record, a) for a in all_raw]
        for fut in as_completed(futures):
            rec = fut.result()
            if rec:
                records.append(rec)
 
    records.sort(key=lambda r: r["h_index"], reverse=True)
    _write_authors_csv(records, AUTHORS_CSV)
    print(f"Saved {len(records):,} author records to '{AUTHORS_CSV}'")
    return records
 
 
# =============================================================================
# EMBEDDING & SIMILARITY
# =============================================================================
 
def _batch_get_vectors(abstracts, embedding_cache):
    """
    Return embedding vectors for a list of abstract dicts, using the shared
    embedding cache to avoid re-encoding texts seen in previous runs.
 
    All cache misses are encoded in a single batched forward pass via
    SentenceTransformer.encode(), which is significantly faster than encoding
    one abstract at a time.
    """
    texts  = [item["abstract"] for item in abstracts]
    misses = [t for t in texts if t not in embedding_cache]
 
    if misses:
        try:
            vecs = _EMBED_MODEL.encode(
                misses,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for text, vec in zip(misses, vecs):
                embedding_cache[text] = vec.tolist()
        except Exception as e:
            print(f"  Embedding error: {e}")
            for text in misses:
                embedding_cache[text] = None
 
    return [v for v in (embedding_cache.get(t) for t in texts) if v is not None]
 
 
def _safe_filename(name):
    """Replace characters that are illegal or awkward in filenames."""
    return re.sub(r'[\\/:*?"<>|]', "-", name).replace(" ", "_")
 
 
def _npz_path(author_name, author_id):
    return os.path.join(SIMILARITY_DIR, f"{_safe_filename(author_name)}_{author_id}.npz")
 
 
def _sim_csv_path(author_name, author_id):
    return os.path.join(SIMILARITY_DIR, f"{_safe_filename(author_name)}_{author_id}.csv")
 
 
def calculate_similarity_matrix(abstracts_list, embedding_cache,
                                 author_name=None, author_id=None):
    """
    Build an upper-triangle cosine similarity matrix for one author's abstracts.
 
    Caching strategy:
      - First call: embeds abstracts, computes matrix, saves a compressed .npz
        (binary cache) and a human-readable pairwise .csv inside SIMILARITY_DIR.
      - Subsequent calls: loads values from the .npz -- no embedding or
        matrix computation performed.
 
    Args:
        abstracts_list (list[dict]): [{'abstract': str}, ...]
        embedding_cache (dict):      Shared text->vector cache (mutated in place).
        author_name (str|None):      Used to build output filenames.
        author_id   (str|None):      Used to build output filenames.
 
    Returns:
        pd.DataFrame | None: N x N DataFrame (upper triangle populated, rest
                             zero), or None if fewer than MIN_ABSTRACTS valid
                             abstracts exist.
    """
    os.makedirs(SIMILARITY_DIR, exist_ok=True)
 
    save  = bool(author_name and author_id)
    npz   = _npz_path(author_name, author_id)    if save else None
    csvsm = _sim_csv_path(author_name, author_id) if save else None
 
    # Cache hit -- reconstruct matrix from .npz
    if save and os.path.exists(npz):
        d   = np.load(npz)
        n   = int(d["n"])
        sim = np.zeros((n, n), dtype=np.float64)
        sim[d["rows"], d["cols"]] = d["vals"]
        return pd.DataFrame(sim)
 
    # Cache miss -- compute from scratch
    valid = [a for a in abstracts_list if a.get("abstract", "").strip()]
    if len(valid) < MIN_ABSTRACTS:
        return None
 
    vectors = _batch_get_vectors(valid, embedding_cache)
    if len(vectors) < MIN_ABSTRACTS:
        return None
 
    n       = len(vectors)
    mat     = np.array(vectors, dtype=np.float32)
    norms   = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normed  = mat / norms
    sim_f32 = np.triu(normed @ normed.T, k=1)
 
    rows, cols = np.triu_indices(n, k=1)
    vals       = sim_f32[rows, cols]
 
    if save:
        np.savez_compressed(npz, rows=rows, cols=cols, vals=vals, n=np.int32(n))
        with open(csvsm, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["abstract_i", "abstract_j", "cosine_similarity"])
            for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
                w.writerow([r, c, round(float(v), SIMILARITY_DECIMAL_PLACES)])
 
    return pd.DataFrame(sim_f32.astype(np.float64))
 
 
# =============================================================================
# MAIN PIPELINE
# =============================================================================
 
def _process_author(author_id, cached_h_index=None, author_name=None):
    """Fetch one author's h-index and abstracts. Runs inside a thread."""
    h_index, abstracts = get_author_data(author_id, cached_h_index=cached_h_index)
    if h_index is None:
        print(f"  Skipping {author_id}: could not retrieve author data")
        return None
    if not abstracts:
        print(f"  Skipping {author_id}: no abstracts found")
        return None
    return author_id, author_name or author_id, h_index, abstracts
 
 
def plot_similarities_vs_h_index(author_ids, h_index_cache=None,
                                  name_cache=None, mcgill_records=None):
    """
    Compute pairwise cosine self-similarities for each author, write the stats
    (abstract_count, avg_self_similarity, std_self_similarity) back to
    AUTHORS_CSV, and plot similarity vs h-index.
 
    Authors whose avg_self_similarity is already populated in mcgill_records
    are skipped entirely -- no API calls or embedding work is done for them.
 
    Args:
        author_ids (list[str]):           OpenAlex author IDs to process.
        h_index_cache (dict, optional):   author_id -> h_index (from CSV).
        name_cache (dict, optional):      author_id -> display name (from CSV).
        mcgill_records (list[dict]|None): Live record list; stats are written
                                          back in-place then the CSV is rewritten.
    """
    h_index_cache = h_index_cache or {}
    name_cache    = name_cache    or {}
    record_by_id  = {r["author_id"]: r for r in (mcgill_records or [])}
 
    # Skip authors whose similarity stats are already recorded in the CSV
    ids_to_process = [
        aid for aid in author_ids
        if record_by_id.get(aid, {}).get("avg_self_similarity") in (None, "")
    ]
    skipped = len(author_ids) - len(ids_to_process)
    if skipped:
        print(f"Skipping {skipped:,} authors already processed in CSV")
 
    embedding_cache = _load_embedding_cache()
 
    # Fetch works concurrently
    author_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(
                _process_author, aid,
                h_index_cache.get(aid),
                name_cache.get(aid),
            ): aid
            for aid in ids_to_process
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                author_results.append(result)
 
    # Compute similarities serially (embedding model is not thread-safe)
    all_h, all_sims, avg_h, avg_sims = [], [], [], []
 
    for author_id, author_name, h_index, abstracts in author_results:
        # Deduplicate abstracts before computing -- identical texts would
        # inflate similarity scores and waste embedding computation.
        unique_abstracts = list({
            a["abstract"] for a in abstracts if a.get("abstract", "").strip()
        })
        abstract_count = len(unique_abstracts)
 
        # Record abstract_count regardless of whether the matrix can be built
        if author_id in record_by_id:
            record_by_id[author_id]["abstract_count"] = abstract_count
 
        sim_df = calculate_similarity_matrix(
            [{"abstract": t} for t in unique_abstracts],
            embedding_cache,
            author_name=author_name,
            author_id=author_id,
        )
 
        if sim_df is None or sim_df.empty:
            print(f"  Skipping {author_id}: fewer than {MIN_ABSTRACTS} abstracts")
            continue
 
        n          = len(sim_df)
        rows, cols = np.triu_indices(n, k=1)
        sims       = sim_df.values[rows, cols]
        avg_sim    = float(np.mean(sims))
        std_sim    = float(np.std(sims))
 
        if author_id in record_by_id:
            record_by_id[author_id]["avg_self_similarity"] = round(
                avg_sim, SIMILARITY_DECIMAL_PLACES
            )
            record_by_id[author_id]["std_self_similarity"] = round(
                std_sim, SIMILARITY_DECIMAL_PLACES
            )
 
        all_h.extend([h_index] * len(sims))
        all_sims.extend(sims.tolist())
        avg_h.append(h_index)
        avg_sims.append(avg_sim)
 
    # Persist embedding cache and rewrite the authors CSV with updated stats
    _save_embedding_cache(embedding_cache)
 
    if mcgill_records and author_results:
        _write_authors_csv(mcgill_records, AUTHORS_CSV)
        print(f"Updated stats written to '{AUTHORS_CSV}'")
 
    if not all_sims:
        print("No new similarity data to plot.")
        return
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
 
    ax1.scatter(all_h, all_sims, alpha=0.7)
    ax1.set_xlabel("H-Index")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Individual Work Similarities vs H-Index")
    ax1.grid(True, alpha=0.3)
 
    ax2.scatter(avg_h, avg_sims, color="red", alpha=0.7)
    ax2.set_xlabel("H-Index")
    ax2.set_ylabel("Average Cosine Similarity")
    ax2.set_title("Average Similarity vs H-Index")
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(RESULTS_PLOT_PATH)
    plt.show()
 
 
# =============================================================================
# ENTRY POINT
# =============================================================================
 
mcgill_records = get_mcgill_authors()
 
if mcgill_records:
    LIST_OF_AUTHOR_IDS = [r["author_id"] for r in mcgill_records]
    H_INDEX_CACHE      = {r["author_id"]: r["h_index"] for r in mcgill_records}
    NAME_CACHE         = {r["author_id"]: r["name"]    for r in mcgill_records}
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
)
 