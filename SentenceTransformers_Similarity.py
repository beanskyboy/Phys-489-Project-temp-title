import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
OPENALEX_API_KEY = "tTekBojph8alEiFnymyAwn"
OPENALEX_MAILTO  = "benjamin.collins3@mail.mcgill.ca"

# --- Institution filter ------------------------------------------------------
INSTITUTION_ID = "I5023651"

# --- Output files & folders --------------------------------------------------
# AUTHORS_CSV: path to the main CSV that stores one row per author.
AUTHORS_CSV = "mcgill_authors_SentenceTransformers.csv"

# SIMILARITY_DIR: folder where per-author pairwise similarity CSVs are stored.
#   Each author gets one file: {Name}_{ID}.csv — human-readable pairwise table
#   with columns [title_i, title_j, cosine_similarity].
SIMILARITY_DIR = "Author_self_similarity_SentenceTransformers"

# EMBEDDING_CACHE_PATH: JSON file that caches {AuthorName}_{WorkTitle} ->
#   embedding-vector mappings so works already embedded in a previous run are
#   not re-sent to the model.  Delete to force full re-embedding.
EMBEDDING_CACHE_PATH = "embedding_cache_SentenceTransformers.json"

# RESULTS_PLOT_PATH: base path for output figures.  The extension is stripped
#   and descriptive suffixes are added for each individual figure saved:
#     {base}_fig1_violin_hindex.png
#     {base}_fig2_violin_abstracts.png
#     {base}_fig3_violin_coverage.png
#     {base}_fig4_scatter.png
#     {base}_fig5_distributions.png
RESULTS_PLOT_PATH = "results_SentenceTransformers.png"

# --- Embedding model ---------------------------------------------------------
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
EMBEDDING_BATCH_SIZE = 64

# --- Similarity computation --------------------------------------------------
# MIN_ABSTRACTS: minimum number of unique works (with title + abstract) an
#   author must have to be included in the similarity analysis.
MIN_ABSTRACTS = 3

SIMILARITY_DECIMAL_PLACES = 6

# --- Parallelism -------------------------------------------------------------
MAX_WORKERS = 8

# --- OpenAlex pagination -----------------------------------------------------
WORKS_PAGE_SIZE   = 200
AUTHORS_PAGE_SIZE = 200

# MAX_FIELDS_OF_STUDY: how many distinct field-of-study labels to record per author.
MAX_FIELDS_OF_STUDY = 4

# REQUIRED_FIELDS: set of field-of-study names an author must have at least one
#   match against to be included in the analysis. Case-insensitive substring.
#   Set to an empty set to disable filtering: REQUIRED_FIELDS = set()
REQUIRED_FIELDS = {
    "Physics",
    #"Astronomy",
    #"Chemistry",
    #"Mathematics",
    #"Computer Science",
    #"Engineering",
    #"Earth and Planetary Sciences",
    #"Materials Science",
    #"Biology",
}

# PROCESSED_AUTHORS_LOG: JSON file recording IDs of every author already
#   handled by the similarity pipeline. Delete to reprocess all from scratch.
PROCESSED_AUTHORS_LOG = "processed_authors.json"

# =============================================================================
# END OF CONFIGURATION -- do not edit below unless changing functionality
# =============================================================================

_OPENALEX_PARAMS = {"api_key": OPENALEX_API_KEY, "mailto": OPENALEX_MAILTO}

SESSION = requests.Session()
SESSION.params.update(_OPENALEX_PARAMS)

_EMBED_MODEL = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)


# =============================================================================
# EMBEDDING CACHE I/O
# Cache key format: "{author_name}_{work_title}"
# =============================================================================

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


def _embed_cache_key(author_name, title):
    """Return the cache key for a given author + work title."""
    return f"{author_name}_{title}"


# =============================================================================
# PROCESSED-AUTHORS LOG I/O
# =============================================================================

def _load_processed_authors():
    if not os.path.exists(PROCESSED_AUTHORS_LOG):
        return set()
    try:
        with open(PROCESSED_AUTHORS_LOG, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, OSError):
        return set()


def _save_processed_authors(author_ids):
    with open(PROCESSED_AUTHORS_LOG, "w", encoding="utf-8") as f:
        json.dump(sorted(author_ids), f)


# Single source of truth for AUTHORS_CSV column names.
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
    Fetch an author's h-index and list of works (with abstract, title, DOI)
    from OpenAlex.

    Deduplication: works sharing the same title (case-insensitive) or the same
    DOI are deduplicated — only the first occurrence is kept.  Works without a
    title are excluded because they cannot be reliably keyed or labelled.

    Returns:
        (h_index, items) where items is a list of dicts:
            {"abstract": str, "title": str, "doi": str}
        Returns (None, None) on API failure.
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
        f"&select=abstract_inverted_index,title,doi"
    )
    while url:
        data = _get_json(url)
        if not data:
            break
        works.extend(data.get("results", []))
        url = data.get("meta", {}).get("next")

    items        = []
    seen_titles  = set()   # normalised lower-case title
    seen_dois    = set()   # normalised lower-case DOI

    for work in works:
        title = (work.get("title") or "").strip()
        doi   = (work.get("doi")   or "").strip().lower()
        idx   = work.get("abstract_inverted_index")

        # Require a title so we can key the cache and label the CSV
        if not title:
            continue

        title_key = title.lower()

        # Deduplicate by title and/or DOI
        if title_key in seen_titles:
            continue
        if doi and doi in seen_dois:
            continue

        if idx:
            text = _invert_abstract_index(idx)
            if text:
                items.append({"abstract": text, "title": title, "doi": doi})
                seen_titles.add(title_key)
                if doi:
                    seen_dois.add(doi)

    return h_index, items


# =============================================================================
# INSTITUTION AUTHOR DISCOVERY
# =============================================================================

def _author_matches_field_filter(author):
    if not REQUIRED_FIELDS:
        return True
    required_lower = {f.lower() for f in REQUIRED_FIELDS}
    for topic_entry in author.get("topics", []):
        field_name = (
            topic_entry.get("field", {}).get("display_name", "")
            or topic_entry.get("subfield", {}).get("display_name", "")
        ).lower()
        if any(req in field_name for req in required_lower):
            return True
    return False


def _fetch_institution_author_page(cursor):
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
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_AUTHORS_CSV_FIELDS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def get_mcgill_authors(refresh=False):
    """
    Return author records for the configured INSTITUTION_ID, filtered by
    REQUIRED_FIELDS, using AUTHORS_CSV as a persistent cache.
    """
    processed = _load_processed_authors()

    if not refresh and os.path.exists(AUTHORS_CSV):
        records = _load_csv_records(AUTHORS_CSV)
        already_done = sum(1 for r in records if r["author_id"] in processed)
        print(
            f"Loaded {len(records):,} authors from cache '{AUTHORS_CSV}' "
            f"({already_done:,} already processed, "
            f"pass refresh=True to re-fetch)"
        )
        return records

    print(f"Fetching author list for institution {INSTITUTION_ID} from OpenAlex ...")

    all_raw       = []
    cursor        = "*"
    total_seen    = 0
    field_skipped = 0

    while cursor:
        results, cursor = _fetch_institution_author_page(cursor)
        if not results:
            break
        total_seen += len(results)
        matching = [a for a in results if _author_matches_field_filter(a)]
        field_skipped += len(results) - len(matching)
        all_raw.extend(matching)
        print(
            f"  ... {total_seen:,} authors scanned, "
            f"{len(all_raw):,} match field filter",
            end="\r",
        )

    print(
        f"\nScanned {total_seen:,} authors total; "
        f"{len(all_raw):,} match field filter "
        f"({field_skipped:,} discarded)."
    )

    if not all_raw:
        print("No authors found -- check INSTITUTION_ID, REQUIRED_FIELDS, or network.")
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

def _batch_get_vectors(items, author_name, embedding_cache):
    """
    Return a list of (title, vector) pairs for the given work items.

    Cache key: "{author_name}_{title}" — unique per author–work combination,
    avoiding collisions between works with the same title across authors.

    All cache misses are encoded in a single batched forward pass.
    Items whose embedding fails or is None are silently dropped from output.

    Args:
        items          (list[dict]): [{"abstract": str, "title": str, ...}]
        author_name    (str):        Used to namespace the cache keys.
        embedding_cache (dict):      Shared cache (mutated in-place).

    Returns:
        list[tuple[str, list[float]]]: [(title, vector), ...]
    """
    def _key(item):
        return _embed_cache_key(author_name, item["title"])

    misses = [item for item in items if _key(item) not in embedding_cache]

    if misses:
        texts = [item["abstract"] for item in misses]
        try:
            vecs = _EMBED_MODEL.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for item, vec in zip(misses, vecs):
                embedding_cache[_key(item)] = vec.tolist()
        except Exception as e:
            print(f"  Embedding error: {e}")
            for item in misses:
                embedding_cache[_key(item)] = None

    result = []
    for item in items:
        vec = embedding_cache.get(_key(item))
        if vec is not None:
            result.append((item["title"], vec))
    return result


def _safe_filename(name):
    """Replace characters that are illegal or awkward in filenames."""
    return re.sub(r'[\\/:*?"<>|]', "-", name).replace(" ", "_")


def _sim_csv_path(author_name, author_id):
    return os.path.join(
        SIMILARITY_DIR,
        f"{_safe_filename(author_name)}_{author_id}.csv"
    )


def calculate_similarity_matrix(items, embedding_cache,
                                 author_name=None, author_id=None):
    """
    Build an upper-triangle cosine similarity matrix for one author's works.

    The pairwise CSV (saved to SIMILARITY_DIR) labels each pair by work title
    rather than index, making it directly human-readable.  No binary .npz cache
    is written — the embedding cache handles the expensive re-computation.

    Args:
        items          (list[dict]): [{"abstract": str, "title": str, ...}]
        embedding_cache (dict):      Shared text->vector cache (mutated in place).
        author_name    (str|None):   Used for cache keys and output filenames.
        author_id      (str|None):   Used for output filenames.

    Returns:
        pd.DataFrame | None: N×N DataFrame (upper triangle populated, rest zero),
                             or None if fewer than MIN_ABSTRACTS valid items exist.
    """
    os.makedirs(SIMILARITY_DIR, exist_ok=True)

    save  = bool(author_name and author_id)
    csvsm = _sim_csv_path(author_name, author_id) if save else None

    valid = [
        item for item in items
        if item.get("abstract", "").strip() and item.get("title", "").strip()
    ]
    if len(valid) < MIN_ABSTRACTS:
        return None

    title_vec_pairs = _batch_get_vectors(valid, author_name, embedding_cache)
    if len(title_vec_pairs) < MIN_ABSTRACTS:
        return None

    titles  = [tv[0] for tv in title_vec_pairs]
    vectors = [tv[1] for tv in title_vec_pairs]

    n      = len(vectors)
    mat    = np.array(vectors, dtype=np.float32)
    norms  = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normed = mat / norms
    sim_f32 = np.triu(normed @ normed.T, k=1)

    rows, cols = np.triu_indices(n, k=1)
    vals       = sim_f32[rows, cols]

    if save:
        with open(csvsm, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["title_i", "title_j", "cosine_similarity"])
            for r, c, v in zip(rows.tolist(), cols.tolist(), vals.tolist()):
                w.writerow([
                    titles[r],
                    titles[c],
                    round(float(v), SIMILARITY_DECIMAL_PLACES),
                ])

    return pd.DataFrame(sim_f32.astype(np.float64))


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def _violin_ax(ax, grouped_data, bin_labels, ylabel, xlabel, title):
    """
    Render a violin + individual-point strip combo on the given Axes.

    Groups with fewer than 2 points cannot produce a violin and receive only
    the strip plot.  Sample counts are shown below each bin label.
    """
    positions = list(range(len(grouped_data)))

    vp_idx  = [i for i, g in enumerate(grouped_data) if len(g) >= 2]
    vp_data = [grouped_data[i] for i in vp_idx]

    if vp_data:
        parts = ax.violinplot(vp_data, positions=vp_idx,
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0")
            pc.set_alpha(0.55)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("#2d4a7a")
                parts[key].set_linewidth(1.5)

    # Strip overlay (jittered)
    rng = np.random.default_rng(42)
    for i, group in enumerate(grouped_data):
        if group:
            jitter = rng.uniform(-0.07, 0.07, size=len(group))
            ax.scatter(
                np.full(len(group), i, dtype=float) + jitter,
                group, s=20, alpha=0.55, color="#c0392b", zorder=3,
            )

    ax.set_xticks(positions)
    tick_labels = [f"{lbl}\n(n={len(g)})"
                   for lbl, g in zip(bin_labels, grouped_data)]
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=6)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")


def _scatter_with_trend(ax, x, y, color, xlabel, ylabel, title,
                         c_arr=None, cmap="viridis", clabel=None):
    """
    Scatter plot with a linear trend line and Pearson r annotation.
    Optionally colour-codes points by a third variable c_arr.
    """
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    mask  = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]

    if c_arr is not None:
        c_arr = np.array(c_arr, dtype=float)[mask]
        sc = ax.scatter(x_arr, y_arr, c=c_arr, cmap=cmap,
                        alpha=0.65, s=25, edgecolors="none")
        plt.colorbar(sc, ax=ax, label=clabel or "")
    else:
        ax.scatter(x_arr, y_arr, color=color, alpha=0.65, s=25, edgecolors="none")

    if len(x_arr) >= 2:
        z = np.polyfit(x_arr, y_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
        ax.plot(x_line, p(x_line), "r--", linewidth=1.5, alpha=0.8)
        r = np.corrcoef(x_arr, y_arr)[0, 1]
        ax.text(0.97, 0.05, f"r = {r:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color="darkred",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=6)
    ax.grid(True, alpha=0.3, linestyle="--")


def _bin_data(values, group_values, bin_edges, bin_labels):
    """
    Group `group_values` according to which bin of `bin_edges` each
    corresponding entry of `values` falls into.

    Returns a list of lists, one per bin (including empty bins).
    """
    groups = [[] for _ in bin_labels]
    for val, gval in zip(values, group_values):
        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if lo <= val < hi:
                groups[i].append(gval)
                break
        else:
            # Catch the upper boundary in the last bin
            if val == bin_edges[-1]:
                groups[-1].append(gval)
    return groups


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def _process_author(author_id, cached_h_index=None, author_name=None):
    """Fetch one author's h-index and work items.  Runs inside a thread."""
    h_index, items = get_author_data(author_id, cached_h_index=cached_h_index)
    if h_index is None:
        print(f"  Skipping {author_id}: could not retrieve author data")
        return None
    if not items:
        print(f"  Skipping {author_id}: no works with title+abstract found")
        return None
    return author_id, author_name or author_id, h_index, items


def plot_figures(author_ids, h_index_cache=None, name_cache=None,
                 mcgill_records=None):
    """
    Compute pairwise cosine self-similarities for each author, persist stats to
    AUTHORS_CSV, and produce a suite of five individual analysis figures:

        Fig 1 — Violin: avg self-similarity per h-index bin
        Fig 2 — Violin: avg self-similarity per abstract-count bin
        Fig 3 — Violin: avg self-similarity per publication-coverage bin
        Fig 4 — Scatter analysis grid (4 panels)
        Fig 5 — Distributional analysis (histogram + correlation heatmap)

    All figures use data from the full AUTHORS_CSV (not just the current run)
    so that repeated partial runs accumulate into complete plots.

    Args:
        author_ids     (list[str]):      OpenAlex author IDs to process.
        h_index_cache  (dict, optional): author_id -> h_index (from CSV cache).
        name_cache     (dict, optional): author_id -> display name (from CSV).
        mcgill_records (list[dict]|None): Live record list; updated in-place.
    """
    h_index_cache = h_index_cache or {}
    name_cache    = name_cache    or {}
    record_by_id  = {r["author_id"]: r for r in (mcgill_records or [])}

    # Skip authors already processed (present in log or already have scores)
    processed      = _load_processed_authors()
    ids_to_process = [
        aid for aid in author_ids
        if aid not in processed
        and record_by_id.get(aid, {}).get("avg_self_similarity") in (None, "")
    ]
    skipped = len(author_ids) - len(ids_to_process)
    if skipped:
        print(f"Skipping {skipped:,} authors (already processed)")

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

    # -------------------------------------------------------------------
    # Compute similarity matrices serially (model is not thread-safe)
    # -------------------------------------------------------------------
    for author_id, author_name, h_index, items in author_results:
        abstract_count = len(items)   # already deduplicated in get_author_data

        if author_id in record_by_id:
            record_by_id[author_id]["abstract_count"] = abstract_count

        

        sim_df = calculate_similarity_matrix(
            items,
            embedding_cache,
            author_name=author_name,
            author_id=author_id,
        )

        if sim_df is None or sim_df.empty:
            print(f"  Skipping {author_id}: fewer than {MIN_ABSTRACTS} valid works")
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
        # Mark processed immediately so a mid-run crash doesn't redo this author
        processed.add(author_id)
        _save_processed_authors(processed)


    # Persist caches and updated CSV
    _save_embedding_cache(embedding_cache)

    if mcgill_records and author_results:
        _write_authors_csv(mcgill_records, AUTHORS_CSV)
        print(f"Updated stats written to '{AUTHORS_CSV}'")

    # -------------------------------------------------------------------
    # Build plot dataset from the full CSV (all runs, not just this one)
    # -------------------------------------------------------------------
    if not os.path.exists(AUTHORS_CSV):
        print("No similarity data to plot.")
        return

    all_records = _load_csv_records(AUTHORS_CSV)
    plot_records = [
        r for r in all_records
        if r.get("avg_self_similarity") is not None
        and r.get("abstract_count") is not None
    ]

    if not plot_records:
        print("No similarity data available for plotting yet.")
        return

    print(f"\nGenerating figures from {len(plot_records):,} authors with similarity scores ...")

    h_idx       = np.array([r["h_index"]             for r in plot_records], dtype=float)
    avg_sims    = np.array([r["avg_self_similarity"]  for r in plot_records], dtype=float)
    std_sims    = np.array([r["std_self_similarity"]  for r in plot_records], dtype=float)
    abs_counts  = np.array([r["abstract_count"]       for r in plot_records], dtype=float)
    pub_counts  = np.array([r["publication_count"]    for r in plot_records], dtype=float)
    cite_counts = np.array([r["citation_count"]       for r in plot_records], dtype=float)

    # Coverage = abstracts with title / total publications (%)
    coverage = np.where(pub_counts > 0, abs_counts / pub_counts * 100.0, 0.0)

    base_path = os.path.splitext(RESULTS_PLOT_PATH)[0]

    # ===================================================================
    # Figure 1 — Violin: avg self-similarity vs h-index bins
    # ===================================================================
    h_edges  = [0, 10, 20, 30, 40, float("inf")]
    h_labels = ["0–9", "10–19", "20–29", "30–39", "40+"]
    h_groups = _bin_data(h_idx, avg_sims.tolist(), h_edges, h_labels)

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    _violin_ax(
        ax1, h_groups, h_labels,
        ylabel="Avg Self-Similarity (cosine)",
        xlabel="H-index range",
        title="Self-Similarity Distribution by H-index",
    )
    fig1.tight_layout()
    fig1.savefig(f"{base_path}_fig1_violin_hindex.png", dpi=150)
    plt.close(fig1)
    print(f"  Saved: {base_path}_fig1_violin_hindex.png")

    # ===================================================================
    # Figure 2 — Violin: avg self-similarity vs abstract count bins
    # ===================================================================
    ac_edges  = [MIN_ABSTRACTS, 10, 25, 50, 100, float("inf")]
    ac_labels = [
        f"{MIN_ABSTRACTS}–9", "10–24", "25–49", "50–99", "100+"
    ]
    ac_groups = _bin_data(abs_counts, avg_sims.tolist(), ac_edges, ac_labels)

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    _violin_ax(
        ax2, ac_groups, ac_labels,
        ylabel="Avg Self-Similarity (cosine)",
        xlabel="Number of abstracts used",
        title="Self-Similarity Distribution by Abstract Count",
    )
    fig2.tight_layout()
    fig2.savefig(f"{base_path}_fig2_violin_abstracts.png", dpi=150)
    plt.close(fig2)
    print(f"  Saved: {base_path}_fig2_violin_abstracts.png")

    # ===================================================================
    # Figure 3 — Violin: avg self-similarity vs coverage percentage bins
    # ===================================================================
    cov_edges  = [0, 25, 50, 75, 101]
    cov_labels = ["0–24 %", "25–49 %", "50–74 %", "75–100 %"]
    cov_groups = _bin_data(coverage, avg_sims.tolist(), cov_edges, cov_labels)

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    _violin_ax(
        ax3, cov_groups, cov_labels,
        ylabel="Avg Self-Similarity (cosine)",
        xlabel="Abstract coverage (% of publications)",
        title="Self-Similarity Distribution by Publication Coverage",
    )
    fig3.tight_layout()
    fig3.savefig(f"{base_path}_fig3_violin_coverage.png", dpi=150)
    plt.close(fig3)
    print(f"  Saved: {base_path}_fig3_violin_coverage.png")

    # ===================================================================
    # Figure 4 — Scatter analysis grid (2 × 2)
    #   [0,0] avg_sim vs h_index
    #   [0,1] avg_sim vs citation count
    #   [1,0] std_sim vs avg_sim  (coloured by h_index)
    #   [1,1] coverage % vs avg_sim  (coloured by abstract count)
    # ===================================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))

    _scatter_with_trend(
        axes4[0, 0], h_idx, avg_sims, color="#2196F3",
        xlabel="H-index",
        ylabel="Avg Self-Similarity",
        title="Avg Self-Similarity vs H-index",
    )
    _scatter_with_trend(
        axes4[0, 1], cite_counts, avg_sims, color="#4CAF50",
        xlabel="Total Citation Count",
        ylabel="Avg Self-Similarity",
        title="Avg Self-Similarity vs Citation Count",
    )
    _scatter_with_trend(
        axes4[1, 0], avg_sims, std_sims, color=None,
        xlabel="Avg Self-Similarity",
        ylabel="Std Self-Similarity",
        title="Variability vs Mean Self-Similarity\n(colour = H-index)",
        c_arr=h_idx, cmap="plasma", clabel="H-index",
    )
    _scatter_with_trend(
        axes4[1, 1], coverage, avg_sims, color=None,
        xlabel="Abstract Coverage (%)",
        ylabel="Avg Self-Similarity",
        title="Self-Similarity vs Publication Coverage\n(colour = abstract count)",
        c_arr=abs_counts, cmap="YlOrRd", clabel="Abstract count",
    )

    fig4.suptitle("Self-Similarity Scatter Analysis", fontsize=13, y=1.01)
    fig4.tight_layout()
    fig4.savefig(f"{base_path}_fig4_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: {base_path}_fig4_scatter.png")

    # ===================================================================
    # Figure 5 — Distributions + correlation heatmap
    #   Left : Histogram of avg self-similarity (all authors)
    #   Right: Correlation heatmap across key bibliometric variables
    # ===================================================================
    fig5 = plt.figure(figsize=(13, 5))
    gs   = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2], figure=fig5)

    # -- Histogram
    ax5a = fig5.add_subplot(gs[0])
    n_bins = max(10, int(np.sqrt(len(avg_sims))))
    ax5a.hist(avg_sims, bins=n_bins, color="#4C72B0", edgecolor="white",
              alpha=0.85, density=True)
    ax5a.axvline(float(np.median(avg_sims)), color="red",
                 linestyle="--", linewidth=1.5,
                 label=f"median = {np.median(avg_sims):.3f}")
    ax5a.axvline(float(np.mean(avg_sims)), color="orange",
                 linestyle=":", linewidth=1.5,
                 label=f"mean   = {np.mean(avg_sims):.3f}")
    ax5a.set_xlabel("Avg Self-Similarity", fontsize=10)
    ax5a.set_ylabel("Density", fontsize=10)
    ax5a.set_title("Distribution of Avg Self-Similarity", fontsize=11)
    ax5a.legend(fontsize=8)
    ax5a.grid(True, alpha=0.3, linestyle="--")

    # -- Correlation heatmap
    ax5b = fig5.add_subplot(gs[1])
    df_corr = pd.DataFrame({
        "H-index":        h_idx,
        "Citations":      cite_counts,
        "Pub count":      pub_counts,
        "Abstract count": abs_counts,
        "Coverage (%)":   coverage,
        "Avg similarity": avg_sims,
        "Std similarity": std_sims,
    }).corr(method="pearson")

    im = ax5b.imshow(df_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax5b, label="Pearson r")
    labels = df_corr.columns.tolist()
    ax5b.set_xticks(range(len(labels)))
    ax5b.set_yticks(range(len(labels)))
    ax5b.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax5b.set_yticklabels(labels, fontsize=8)
    ax5b.set_title("Pearson Correlation Heatmap", fontsize=11)

    # Annotate each cell with the r value
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = df_corr.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax5b.text(j, i, f"{val:.2f}", ha="center", va="center",
                      fontsize=7, color=color)

    fig5.tight_layout()
    fig5.savefig(f"{base_path}_fig5_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)
    print(f"  Saved: {base_path}_fig5_distributions.png")

    print("\nAll figures saved.")


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

plot_figures(
    LIST_OF_AUTHOR_IDS,
    h_index_cache=H_INDEX_CACHE,
    name_cache=NAME_CACHE,
    mcgill_records=mcgill_records,
)