import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
import spacy


# INPUT_CSV = r"C:\Users\mauth\PHYS 489\Phys-489-Project-temp-title\Similarity tests.csv"
# OUTPUT_CSV = r"C:\Users\mauth\PHYS 489\Phys-489-Project-temp-title\Similarity tests_spacy_scored(1).csv"
# FIRST_COLUMN = "first_sentence"
# SECOND_COLUMN = "second_sentence"
# MODEL_NAME = "en_core_web_lg"
# SCORE_COLUMN = "spacy_similarity"

INPUT_CSV = r"C:\Users\mauth\PHYS 489\Phys-489-Project-temp-title\Similarity tests.csv"
OUTPUT_CSV = r"C:\Users\mauth\PHYS 489\Phys-489-Project-temp-title\Similarity tests_sentence_transformers_scored.csv"
FIRST_COLUMN = "first_sentence"
SECOND_COLUMN = "second_sentence"
MODEL_NAME = "all-mpnet-base-v2"
# MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
SCORE_COLUMN = "sentence_transformer_similarity"


def main():
    df = pd.read_csv(INPUT_CSV, encoding="latin1")

    df[FIRST_COLUMN] = df[FIRST_COLUMN].astype(str).str.strip()
    df[SECOND_COLUMN] = df[SECOND_COLUMN].astype(str).str.strip()

    model = SentenceTransformer(MODEL_NAME)

    unique_texts = pd.concat([df[FIRST_COLUMN], df[SECOND_COLUMN]]).drop_duplicates().tolist()
    embeddings = model.encode(unique_texts, convert_to_numpy=True)
    text_to_embedding = {
        text: np.array(embedding, dtype=float)
        for text, embedding in zip(unique_texts, embeddings)
    }

    scores = []
    for _, row in df.iterrows():
        vec_a = np.expand_dims(text_to_embedding[row[FIRST_COLUMN]], axis=0)
        vec_b = np.expand_dims(text_to_embedding[row[SECOND_COLUMN]], axis=0)
        scores.append(float(model.similarity(vec_a, vec_b).item()))

    df[SCORE_COLUMN] = scores
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved scored CSV to: {OUTPUT_CSV}")
#Spacy similarity scoring
    # nlp = spacy.load(MODEL_NAME)

    # unique_texts = pd.concat([df[FIRST_COLUMN], df[SECOND_COLUMN]]).drop_duplicates().tolist()
    # docs = list(nlp.pipe(unique_texts))
    # text_to_doc = {text: doc for text, doc in zip(unique_texts, docs)}

    # scores = []
    # for _, row in df.iterrows():
    #     doc_a = nlp(row[FIRST_COLUMN])
    #     doc_b = nlp(row[SECOND_COLUMN])
    #     scores.append(float(doc_a.similarity(doc_b)))

    # df[SCORE_COLUMN] = scores
    # df.to_csv(OUTPUT_CSV, index=False)
    # print(f"Saved scored CSV to: {OUTPUT_CSV}")
if __name__ == "__main__":
    main()
