import ollama
import numpy as np
import pandas as pd
import json
import os
import hashlib

from API_Call_Code import fetch_work, fetch_top_citation_abstracts, extract_abstract



EMBED_MODEL = "embeddinggemma"
EMBED_CACHE_PATH = "embedding_cache.json"
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"

# Pick the backend for run_similarity_test_csv by changing this line.
# TEST_SIMILARITY_BACKEND = "ollama"
TEST_SIMILARITY_BACKEND = "sentence_transformers"

# 1. API Tool: Check if a paper is retracted (Example using a mock API)
# def check_retraction(paper_id):
#     # In a real scenario, you'd call an API like Crossref or Semantic Scholar
#     # For this example, we'll simulate a 'clean' status
#     print(f"[*] Checking retraction status for: {paper_id}")
#     return {"status": "clean"}

# 2. Embedding Helper
def load_embedding_cache(cache_path=EMBED_CACHE_PATH):
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_embedding_cache(cache, cache_path=EMBED_CACHE_PATH):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _embedding_cache_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_vector(text, embedding_cache):
    key = _embedding_cache_key(text)
    if key in embedding_cache:
        return np.array(embedding_cache[key], dtype=float)

    vector = ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]
    embedding_cache[key] = vector
    return np.array(vector, dtype=float)


def cosine_similarity(vec_a, vec_b):
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
    return df


def _extract_embeddings(response):
    if isinstance(response, dict):
        return response["embeddings"]
    return response.embeddings


def get_vectors_for_texts(texts, embedding_cache):
    vectors = []
    missing_texts = []
    missing_keys = []

    for text in texts:
        key = _embedding_cache_key(text)
        if key in embedding_cache:
            vectors.append(np.array(embedding_cache[key], dtype=float))
        else:
            vectors.append(None)
            missing_texts.append(text)
            missing_keys.append(key)

    if missing_texts:
        response = ollama.embed(model=EMBED_MODEL, input=missing_texts)
        embeddings = _extract_embeddings(response)
        for key, embedding in zip(missing_keys, embeddings):
            embedding_cache[key] = embedding

        missing_iter = iter(
            np.array(embedding_cache[key], dtype=float) for key in missing_keys
        )
        vectors = [next(missing_iter) if vec is None else vec for vec in vectors]

    return vectors





def run_similarity_pipeline(
    file_path,
    top_k=5,
    output_path="dataset_top_k_similarities.csv",
    embedding_cache=None,
):
    if embedding_cache is None:
        embedding_cache = {}

    df = pd.read_csv(file_path, encoding="latin1")
    df = normalize_columns(df)

    required_columns = {"id", "title", "abstract"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(
            f"Missing required columns in {file_path}: {sorted(missing_columns)}"
        )

    working_df = df.dropna(subset=["id", "title", "abstract"]).copy()
    working_df["abstract"] = working_df["abstract"].astype(str).str.strip()
    working_df = working_df[working_df["abstract"] != ""].reset_index(drop=True)

    if len(working_df) < 2:
        raise ValueError("Need at least 2 papers with non-empty abstracts.")

    top_k = max(1, min(top_k, len(working_df) - 1))

    texts = working_df["abstract"].tolist()
    vectors = get_vectors_for_texts(texts, embedding_cache)
    matrix = np.vstack(vectors)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    normalized = matrix / norms
    similarity_matrix = normalized @ normalized.T
    np.fill_diagonal(similarity_matrix, -np.inf)

    all_results = []
    doi_present = "doi" in working_df.columns

    for idx, row in working_df.iterrows():
        neighbor_indices = np.argsort(similarity_matrix[idx])[::-1][:top_k]
        neighbors = working_df.iloc[neighbor_indices]
        scores = similarity_matrix[idx, neighbor_indices]

        all_results.append({
            "paper_id": row["id"],
            "paper_doi": row["doi"] if doi_present else None,
            "paper_title": row["title"],
            "paper_abstract": row["abstract"],
            "top_k": top_k,
            "neighbor_ids": json.dumps(neighbors["id"].tolist()),
            "neighbor_dois": json.dumps(
                neighbors["doi"].tolist() if doi_present else [None] * len(neighbors)
            ),
            "neighbor_titles": json.dumps(neighbors["title"].tolist()),
            "neighbor_abstracts": json.dumps(neighbors["abstract"].tolist()),
            "neighbor_similarity_scores": json.dumps(
                [float(score) for score in scores]
            ),
            "average_neighbor_similarity": float(np.mean(scores)),
        })

    output_df = pd.DataFrame(all_results)
    output_df.to_csv(output_path, index=False)
    save_embedding_cache(embedding_cache)
    return output_df


def run_similarity_test_csv(
    file_path,
    output_path=None,
    embedding_cache=None,
    backend=None,
):
    if embedding_cache is None:
        embedding_cache = {}

    df = pd.read_csv(file_path, encoding="latin1")
    df = normalize_columns(df)

    first_col = None
    second_col = None
    score_col = None

    for candidate in ("first_sentence", "word1"):
        if candidate in df.columns:
            first_col = candidate
            break

    for candidate in ("second_sentence", "word2"):
        if candidate in df.columns:
            second_col = candidate
            break

    for candidate in ("similarity score", "similarity_score", "similarity"):
        if candidate in df.columns:
            score_col = candidate
            break

    if first_col is None or second_col is None:
        raise KeyError(
            "Expected sentence columns like "
            "'first_sentence'/'second_sentence' or 'word1'/'word2'."
        )

    if score_col is None:
        score_col = "similarity"
        df[score_col] = np.nan

    working_df = df.dropna(subset=[first_col, second_col]).copy()
    working_df[first_col] = working_df[first_col].astype(str).str.strip()
    working_df[second_col] = working_df[second_col].astype(str).str.strip()
    working_df = working_df[
        (working_df[first_col] != "") & (working_df[second_col] != "")
    ].copy()

    combined_texts = (
        pd.concat([working_df[first_col], working_df[second_col]])
        .drop_duplicates()
        .tolist()
    )
    vectors = get_vectors_with_backend(
        combined_texts,
        embedding_cache=embedding_cache,
        backend=backend,
    )
    text_to_vector = {
        text: vector
        for text, vector in zip(combined_texts, vectors)
    }

    scores = []
    for _, row in working_df.iterrows():
        vec_a = text_to_vector[row[first_col]]
        vec_b = text_to_vector[row[second_col]]
        scores.append(compute_similarity_score(vec_a, vec_b, backend=backend))

    working_df[score_col] = scores
    df.loc[working_df.index, score_col] = working_df[score_col]

    if output_path is None:
        output_path = file_path

    df.to_csv(output_path, index=False)
    if get_similarity_backend_name(backend) == "ollama":
        save_embedding_cache(embedding_cache)
    return df

# 3. Main Logic
def run_research_pipeline(paper_id, top_n=5, embedding_cache=None):
    if embedding_cache is None:
        embedding_cache = {}

    # Step 1: Logic Model (Llama 4) decides to trigger the check
    # paper_data = check_retraction(paper_id)
    
    # if paper_data['status'] == 'retracted':
    #     return {
    #         "paper_id": paper_id,
    #         "error": "Warning: This paper has been retracted. Aborting analysis."
    #     }

    # Step 2: Pull real abstract for target paper
    source_work = fetch_work(paper_id)
    main_abstract = extract_abstract(source_work.get("abstract_inverted_index"))

    if not main_abstract:
        return {
            "paper_id": paper_id,
            "error": "No abstract available for target paper."
        }

    # Step 3: Pull top-N referenced works and keep only those with abstracts
    references = fetch_top_citation_abstracts(paper_id, top_n=top_n)
    references = [r for r in references if r.get("abstract")]

    if not references:
        return {
            "paper_id": paper_id,
            "error": "No referenced abstracts available for similarity analysis."
        }

    if len(references) < top_n:
        return {
            "paper_id": paper_id,
            "error": (
                f"Only found {len(references)} referenced papers with abstracts; "
                f"{top_n} are required for similarity analysis."
            )
        }

    # Step 4: Embed and compute cosine similarities
    main_vec = get_vector(main_abstract, embedding_cache)
    
    similarities = []
    for ref in references:
        ref_vec = get_vector(ref["abstract"], embedding_cache)
        score = cosine_similarity(main_vec, ref_vec)
        similarities.append({
            "reference_id": ref.get("id"),
            "reference_title": ref.get("title"),
            "cited_by_count": ref.get("cited_by_count", 0),
            "cosine_similarity": score
        })

    avg_similarity = float(np.mean([item["cosine_similarity"] for item in similarities]))
    return {
        "paper_id": paper_id,
        "paper_doi": source_work.get("doi"),
        "paper_title": source_work.get("title"),
        "paper_abstract": main_abstract,
        "num_references_compared": len(similarities),
        "average_similarity": avg_similarity,
        "reference_similarities": similarities
    }





def main():
    embedding_cache = load_embedding_cache()
    # file_path = r"C:\Users\mauth\OneDrive\Desktop\School\Winter 2026\PHYS 489\Data\Astronomy and Astrophysics sample.csv"
    # df = pd.read_csv(file_path, encoding="latin1")
    # df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    # if "id" not in df.columns:
    #     raise KeyError(
    #         f"Expected an 'id' column in {file_path}, but found: {list(df.columns)}"
    #     )

    # ids = df["id"].dropna().tolist()[:20]
    # all_results = []
    # for paper_id in ids:
    #     result = run_research_pipeline(paper_id, top_n=5, embedding_cache=embedding_cache)
    #     if "error" in result:
    #         print(f"{paper_id}: {result['error']}")
    #         continue

    #     refs = result["reference_similarities"]
    #     all_results.append({
    #         "paper_id": result["paper_id"],
    #         "paper_doi": result.get("paper_doi"),
    #         "paper_title": result.get("paper_title"),
    #         "paper_abstract": result.get("paper_abstract"),
    #         "num_references_compared": len(refs),
    #         "reference_ids": json.dumps([r.get("reference_id") for r in refs]),
    #         "reference_titles": json.dumps([r.get("reference_title") for r in refs]),
    #         "reference_similarity_scores": json.dumps([r.get("cosine_similarity") for r in refs]),
    #         "average_similarity_score": result["average_similarity"]
    #     })

    #     print(
    #         f"{paper_id} | avg similarity: {result['average_similarity']:.4f} "
    #         f"across {result['num_references_compared']} references"
    #     )

    # output_df = pd.DataFrame(all_results)
    # output_df.to_csv("Astronomy_and_Astrophysics_similarities(1).csv", index=False)
    # save_embedding_cache(embedding_cache)

    # output_df = run_similarity_pipeline(
    #     file_path=file_path,
    #     top_k=5,
    #     output_path="Astronomy_and_Astrophysics_top-k-neighbors.csv",
    #     embedding_cache=embedding_cache,
    # )
    # print(f"Saved {len(output_df)} paper similarity rows.")
  
    output_df = run_similarity_test_csv(
        file_path=r"C:\Users\mauth\PHYS 489\Phys-489-Project-temp-title\Similarity tests.csv",
        output_path=r"C:\Users\mauth\PHYS 489\Phys-489-Project-temp-title\Similarity tests results_sentence-transformer_all-mpnet-base-v2.csv",
        embedding_cache=embedding_cache,
    )
    print(f"Updated {len(output_df)} similarity test rows in {r'C:\Users\mauth\PHYS 489\Phys-489-Project-temp-title\Similarity tests.csv'}.")
  
  
 
if __name__ == "__main__":
    main()
