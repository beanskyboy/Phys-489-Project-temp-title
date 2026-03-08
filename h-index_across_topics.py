
import os
import json
import time
import random
import requests
import argparse
from urllib.parse import quote
from enum import Enum
import Compare_abstracts
import API_Call_Code
import ollama

response = ollama.embed(
    model='embeddinggemma',
    input='The sky is blue because of Rayleigh scattering',
)
print(response.embeddings)

params = {
        "api_key": "tTekBojph8alEiFnymyAwn",
        "mailto": "benjamin.collins3@mail.mcgill.ca"
    }

def get_h_index_from_author_id(author_id):
    """
    Calculate the h-index of an author using their OpenAlex author ID.

    Args:
        author_id (str): The OpenAlex author identifier (e.g., 'A1969205035').

    Returns:
        int: The h-index of the author, or None if not found or error occurred.
    """
    # OpenAlex API endpoint for author by ID
    url = f"https://api.openalex.org/authors/{author_id}"

    

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes

        data = response.json()

        # Check if author was found
        if "h_index" in data["summary_stats"]:
            return data["summary_stats"]["h_index"]
        else:
            return None

    except requests.exceptions.RequestException as e:
        # propagate or log externally; here just return None
        return None
    except KeyError:
        return None

def invert_abstract_index(abstract_index):
    """
    Convert OpenAlex abstract_inverted_index to plain text.

    Args:
        abstract_index (dict): The inverted index dictionary

    Returns:
        str: The reconstructed abstract text, or None if not available
    """
    if not abstract_index:
        return None

    positions = {}
    for word, pos_list in abstract_index.items():
        for pos in pos_list:
            positions[pos] = word

    return ' '.join(positions[i] for i in sorted(positions))

def get_abstracts_from_author_id(author_id):
    """
    Get abstracts of all works by an author using their OpenAlex author ID.

    Args:
        author_id (str): The OpenAlex author identifier (e.g., 'A1969205035')

    Returns:
        list: List of dictionaries with 'title' and 'abstract' for each work, or None if error
    """


    # verify the author exists (fetch once to ensure ID is valid)
    author_url = f"https://api.openalex.org/authors/{author_id}"
    try:
        response = requests.get(author_url, params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    # Now, get all works by this author
    works = []
    url = f"https://api.openalex.org/works?filter=author.id:{author_id}&per-page=100"

    while url:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            works.extend(data.get('results', []))
            url = data.get('meta', {}).get('next')  # Next page URL
        except requests.exceptions.RequestException:
            break

    # Extract titles and abstracts
    abstracts = []
    abstracts_with_text = 0
    for work in works:
        title = work.get('title')
        abstract_index = work.get('abstract_inverted_index')
        abstract = invert_abstract_index(abstract_index) if abstract_index else None
        if abstract:
            abstracts_with_text += 1
        abstracts.append({
            'title': title,
            'abstract': abstract
        })


    return abstracts

def calculate_similarity_matrix(abstracts_list):
    """
    Calculate a similarity matrix for a list of abstracts using embeddings and cosine similarity.

    Args:
        abstracts_list (list): List of dictionaries with 'title' and 'abstract' keys, as returned by get_abstracts_from_author_id

    Returns:
        pd.DataFrame: Similarity matrix with titles as indices and columns
    """
    import Compare_abstracts
    import pandas as pd
    import numpy as np

    # Load embedding cache
    embedding_cache = Compare_abstracts.load_embedding_cache(cache_path="\\embedding_cache.json")

    # Filter abstracts that have text
    valid_abstracts = [item for item in abstracts_list if item.get('abstract') and item['abstract'].strip()]

    if len(valid_abstracts) < 2:
        # not enough abstracts to form a matrix, silence by returning None
        return None

    # Get vectors for each abstract
    vectors = []
    titles = []
    for item in valid_abstracts:
        # to test if it works  print(Compare_abstracts.get_vector(item['abstract'], embedding_cache))
        try:
    
            vec = Compare_abstracts.get_vector(item['abstract'], embedding_cache)
            vectors.append(vec)
        except Exception:
            print("issue")
            continue

    # Save updated cache
    Compare_abstracts.save_embedding_cache(embedding_cache)

    n = len(vectors)
    similarity_matrix = np.zeros((n, n))

    # Compute only upper triangle (excluding diagonal)
    for i in range(n):
        for j in range(i+1, n):
            similarity_matrix[i][j] = Compare_abstracts.cosine_similarity(vectors[i], vectors[j])
            # Symmetric: similarity_matrix[j][i] = similarity_matrix[i][j]  # optional, but since we don't need lower, leave as 0

    # Create DataFrame
    df = pd.DataFrame(similarity_matrix)
    return df

def plot_similarities_vs_h_index(author_id):
    """
    Plot the similarities from the author's works against their h-index.

    Args:
        author_id (str): The OpenAlex author identifier
    """
    import matplotlib.pyplot as plt

    # Get h-index
    h_index = get_h_index_from_author_id(author_id)
    if h_index is None:
        print("Could not retrieve h-index")
        return

    # Get abstracts
    abstracts = get_abstracts_from_author_id(author_id)
    if not abstracts:
        print("Could not retrieve abstracts")
        return

    # Calculate similarity matrix
    sim_df = calculate_similarity_matrix(abstracts)
    if sim_df is None or sim_df.empty:
        print("Could not compute similarity matrix")
        return

    # Extract upper triangle similarities (excluding diagonal)
    similarities = []
    n = len(sim_df)
    for i in range(n):
        for j in range(i+1, n):
            similarities.append(sim_df.iloc[i, j])

    if not similarities:
        print("No similarities to plot")
        return

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter([h_index] * len(similarities), similarities, alpha=0.7)
    plt.xlabel('H-Index')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Work Similarities vs H-Index for author {author_id}\n(H-Index: {h_index})')
    plt.grid(True, alpha=0.3)
    plt.show()

calculate_similarity_matrix(get_abstracts_from_author_id("a5025339678"))




