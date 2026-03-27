import ollama
import numpy as np
import pandas as pd
import json
import os
import hashlib

# from sentence_transformers import SentenceTransformer, util
from API_Call_Code import fetch_work, fetch_top_citation_abstracts, extract_abstract



EMBED_MODEL = "embeddinggemma"
EMBED_CACHE_PATH = "embedding_cache.json"

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
    file_path = r"C:\Users\mauth\OneDrive\Desktop\School\Winter 2026\PHYS 489\Data\Astronomy and Astrophysics sample.csv"
    df = pd.read_csv(file_path, encoding="latin1")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    if "id" not in df.columns:
        raise KeyError(
            f"Expected an 'id' column in {file_path}, but found: {list(df.columns)}"
        )

    ids = df["id"].dropna().tolist()[:20]
    all_results = []
    for paper_id in ids:
        result = run_research_pipeline(paper_id, top_n=5, embedding_cache=embedding_cache)
        if "error" in result:
            print(f"{paper_id}: {result['error']}")
            continue

        refs = result["reference_similarities"]
        all_results.append({
            "paper_id": result["paper_id"],
            "paper_doi": result.get("paper_doi"),
            "paper_title": result.get("paper_title"),
            "paper_abstract": result.get("paper_abstract"),
            "num_references_compared": len(refs),
            "reference_ids": json.dumps([r.get("reference_id") for r in refs]),
            "reference_titles": json.dumps([r.get("reference_title") for r in refs]),
            "reference_similarity_scores": json.dumps([r.get("cosine_similarity") for r in refs]),
            "average_similarity_score": result["average_similarity"]
        })

        print(
            f"{paper_id} | avg similarity: {result['average_similarity']:.4f} "
            f"across {result['num_references_compared']} references"
        )

    output_df = pd.DataFrame(all_results)
    output_df.to_csv("Astronomy_and_Astrophysics_similarities(1).csv", index=False)
    save_embedding_cache(embedding_cache)
  

  
 
if __name__ == "__main__":
    main()
