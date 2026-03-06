import ollama
import requests
import numpy as np
import pandas as pd
import json
import os
import hashlib

from API_Call_Code import fetch_work, fetch_top_citation_abstracts, extract_abstract



EMBED_MODEL = "embeddinggemma"
EMBED_CACHE_PATH = "embedding_cache.json"

# 1. API Tool: Check if a paper is retracted (Example using a mock API)
def check_retraction(paper_id):
    # In a real scenario, you'd call an API like Crossref or Semantic Scholar.
    print(f"[*] Checking retraction status for: {paper_id}")
    return {"status": "clean"}

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
    try:
        source_work = fetch_work(paper_id)
    except requests.exceptions.RequestException as e:
        return {
            "paper_id": paper_id,
            "error": f"Failed to fetch source paper from OpenAlex: {e}"
        }

    main_abstract = extract_abstract(source_work.get("abstract_inverted_index"))

    if not main_abstract:
        return {
            "paper_id": paper_id,
            "error": "No abstract available for target paper."
        }

    # Step 3: Pull top-N referenced works and keep only those with abstracts
    try:
        references = fetch_top_citation_abstracts(paper_id, top_n=top_n)
    except requests.exceptions.RequestException as e:
        return {
            "paper_id": paper_id,
            "error": f"Failed to fetch top referenced works from OpenAlex: {e}"
        }

    references = [r for r in references if r.get("abstract")]

    if not references:
        return {
            "paper_id": paper_id,
            "error": "No referenced abstracts available for similarity analysis."
        }

    # Step 4: Embed and compute cosine similarities
    try:
        main_vec = get_vector(main_abstract, embedding_cache)
    except Exception as e:
        return {
            "paper_id": paper_id,
            "error": f"Failed to embed source abstract: {e}"
        }
    
    similarities = []
    for ref in references:
        try:
            ref_vec = get_vector(ref["abstract"], embedding_cache)
        except Exception as e:
            print(f"{paper_id}: skipping reference {ref.get('id')} due to embedding error: {e}")
            continue

        score = cosine_similarity(main_vec, ref_vec)
        similarities.append({
            "reference_id": ref.get("id"),
            "reference_title": ref.get("title"),
            "cited_by_count": ref.get("cited_by_count", 0),
            "cosine_similarity": score
        })

    if not similarities:
        return {
            "paper_id": paper_id,
            "error": "No valid reference embeddings available for similarity analysis."
        }

    avg_similarity = float(np.mean([item["cosine_similarity"] for item in similarities]))
    return {
        "paper_id": paper_id,
        "paper_title": source_work.get("title"),
        "num_references_compared": len(similarities),
        "average_similarity": avg_similarity,
        "reference_similarities": similarities
    }



def main():
    embedding_cache = load_embedding_cache()
    file_path = r"C:\Users\mauth\OneDrive\Desktop\School\Winter 2026\PHYS 489\papers_batch_1.csv"
    try:
        df = pd.read_csv(file_path,encoding='latin1')
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        return

    if "id" not in df.columns:
        print("CSV does not contain an 'id' column.")
        return

    ids = df['id'].dropna().tolist()[:5]
    for paper_id in ids:
        try:
            result = run_research_pipeline(paper_id, top_n=5, embedding_cache=embedding_cache)
        except Exception as e:
            print(f"{paper_id}: unexpected pipeline error: {e}")
            continue

        if "error" in result:
            print(f"{paper_id}: {result['error']}")
            continue

        print(
            f"{paper_id} | avg similarity: {result['average_similarity']:.4f} "
            f"across {result['num_references_compared']} references"
        )

    try:
        save_embedding_cache(embedding_cache)
    except Exception as e:
        print(f"Failed to save embedding cache: {e}")
  

    
 
if __name__ == "__main__":
    main()
