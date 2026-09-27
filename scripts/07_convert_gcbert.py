import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# ==========================================================
# CONFIGURATION
# ==========================================================
DEFAULT_INPUT = 'C:/Dissertacao/data_bases/04_final/train_validator_final.csv'   # output of script 08
DEFAULT_OUTPUT = 'C:/Dissertacao/data_bases/graphcodebert/embeddings_graphcodebert.csv'
MODEL_NAME = "microsoft/graphcodebert-base"


def pick(df, *names):
    """Returns the first column name that exists (supports the PT and EN versions of the data,
    even mixed in the same file, e.g. Case_ID + Codigo_Snippet + Tem_Fuga_de_Recurso)."""
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"None of the columns {names} found. Available: {list(df.columns)}")


def pair_id(case_id):
    """Ex: 'Case_003_AnkiDroid_SINTETICO_3_BUG' -> 'Case_003_AnkiDroid_SINTETICO_3'"""
    return str(case_id).rsplit('_', 1)[0]


def extract_embedding(code, tokenizer, model, device):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS token (global representation of the code)
    return outputs.last_hidden_state[0, 0, :].cpu().numpy()


def extract_save_embeddings(input_path, output_path, model_name=MODEL_NAME):
    print("Starting Embeddings Extraction (GraphCodeBERT)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # disables dropout, so the same code always yields the same embedding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   -> Running on: {device}")

    try:
        df = pd.read_csv(input_path, sep=';')
        if len(df.columns) == 1:  # wrong separator guessed, retry with comma
            df = pd.read_csv(input_path)
    except Exception:
        df = pd.read_csv(input_path)

    id_col = pick(df, 'Case_ID', 'ID_Caso')
    code_col = pick(df, 'Code_Snippet', 'Codigo_Snippet', 'code_with_bug')
    label_col = pick(df, 'Has_Resource_Leak', 'Tem_Fuga_de_Recurso')

    # Pair_ID to prevent data leakage in the trainer later (script 12 groups by original case on top of this)
    df['Pair_ID'] = df[id_col].apply(pair_id)

    df = df.dropna(subset=[code_col, label_col])
    codes = df[code_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    pair_ids = df['Pair_ID'].tolist()

    print(f"Processing {len(codes)} codes...")

    X_embeddings = []
    for i, code in enumerate(codes):
        if i % 100 == 0 and i > 0:
            print(f"   -> Extracted {i}/{len(codes)}...")
        embedding = extract_embedding(code, tokenizer, model, device)
        X_embeddings.append(embedding)

    print("Saving neural features and IDs to CSV...")
    df_embeddings = pd.DataFrame(X_embeddings)
    df_embeddings.columns = [f"dim_{i}" for i in range(df_embeddings.shape[1])]
    df_embeddings['Has_Resource_Leak'] = labels
    df_embeddings['Pair_ID'] = pair_ids  # script 12 groups on this (via original_case)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_embeddings.to_csv(output_path, index=False)
    print(f"File saved successfully at: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--model-name', default=MODEL_NAME)
    args = parser.parse_args()
    extract_save_embeddings(args.input, args.output, args.model_name)