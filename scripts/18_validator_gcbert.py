import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModel, AutoTokenizer

# ==========================================================
# CONFIGURATION
# ==========================================================
DEFAULT_EMBEDDINGS = 'C:/Dissertacao/data_bases/04_final/graphcodebert/embeddings_graphcodebert.csv'  # script 7
DEFAULT_INPUT_TEMPLATE = 'C:/Dissertacao/data_bases/05_results/tournament_results_holdout_v2_{lang}.csv'
DEFAULT_OUTPUT_TEMPLATE = 'C:/Dissertacao/data_bases/05_results/final_dissertation_report_gcbert_{lang}.csv'
MODEL_NAME = "microsoft/graphcodebert-base"

LLM_COLUMNS = ['fix_gpt', 'fix_claude', 'fix_gemini']
OFFICIAL_NAMES = {'fix_gpt': 'GPT-5.4', 'fix_claude': 'Claude Opus 4.6', 'fix_gemini': 'Gemini 3.1 Pro'}


def pick(df, *names):
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"None of the columns {names} found. Available: {list(df.columns)}")


def is_error(value):
    return not (pd.notna(value) and str(value).strip() != "" and not str(value).strip().startswith("ERROR"))


def train_judge(embeddings_path):
    print("Loading the Judge's knowledge base (training embeddings)...")
    df_base = pd.read_csv(embeddings_path)
    drop_cols = [c for c in ['Has_Resource_Leak', 'Pair_ID', 'ID_Pair'] if c in df_base.columns]
    X_train = df_base.drop(columns=drop_cols).values
    y_train = df_base['Has_Resource_Leak'].values

    print(f"Training the Judge (Random Forest) with {len(X_train)} examples (100% of the training partition)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def load_extractor():
    print("Loading GraphCodeBERT extractor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()  # disables dropout, so the same code always yields the same embedding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"   -> Running on: {device}")
    return tokenizer, model, device


def extract_embedding(code, tokenizer, model, device):
    inputs = tokenizer(str(code), return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0, 0, :].cpu().numpy()


def score_one_language(lang, rf, tokenizer, model, device, input_path, output_path):
    if not os.path.exists(input_path):
        print(f"\n[{lang.upper()}] SKIPPED: file not found at {input_path}")
        return None

    print(f"\n[{lang.upper()}] Reading tournament results from {input_path}...")
    df = pd.read_csv(input_path)
    pick(df, 'Case_ID', 'ID_Caso')  # fail fast if malformed
    total = len(df)
    print(f"   -> {total} bugs loaded")

    summary = {}
    for col in LLM_COLUMNS:
        if col not in df.columns:
            print(f"   WARNING: column '{col}' not found, skipping.")
            continue

        llm_name = OFFICIAL_NAMES.get(col, col)
        print(f"   Extracting embeddings for {llm_name}...")

        error_mask = df[col].apply(is_error)
        n_errors = int(error_mask.sum())

        verdicts = pd.Series(1, index=df.index)  # default: 1 = still has bug (also used for API errors)
        valid_idx = df.index[~error_mask]
        if len(valid_idx) > 0:
            vectors = [extract_embedding(code, tokenizer, model, device) for code in df.loc[valid_idx, col]]
            verdicts.loc[valid_idx] = rf.predict(np.vstack(vectors))

        df[f'verdict_{col}'] = verdicts  # 0 = safe/fixed, 1 = still has bug (or API error)

        fixed = int((verdicts == 0).sum())
        rate = (fixed / total) * 100
        summary[col] = (fixed, n_errors, rate)

    print(f"   FINAL TOURNAMENT SCORE - GraphCodeBERT ({lang.upper()})")
    print("   " + "-" * 46)
    for col, (fixed, n_errors, rate) in summary.items():
        name = OFFICIAL_NAMES.get(col, col)
        print(f"   {name:18s}: {fixed}/{total} fixed ({rate:.2f}%), {n_errors} API errors counted as failures")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, sep=';')
    print(f"   Detailed report saved to '{output_path}'")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['pt', 'en', 'both'], default='both',
                        help="which tournament run(s) to score; 'both' runs pt and en in one go")
    parser.add_argument('--input', default=None,
                        help='overrides the default path (only valid together with --lang pt or --lang en)')
    parser.add_argument('--output', default=None,
                        help='overrides the default path (only valid together with --lang pt or --lang en)')
    parser.add_argument('--embeddings', default=DEFAULT_EMBEDDINGS)
    args = parser.parse_args()

    langs = ['pt', 'en'] if args.lang == 'both' else [args.lang]

    rf = train_judge(args.embeddings)
    tokenizer, model, device = load_extractor()

    all_results = {}
    for lang in langs:
        input_path = args.input or DEFAULT_INPUT_TEMPLATE.format(lang=lang)
        output_path = args.output or DEFAULT_OUTPUT_TEMPLATE.format(lang=lang)
        result = score_one_language(lang, rf, tokenizer, model, device, input_path, output_path)
        if result:
            all_results[lang] = result

    print("\n" + "=" * 60)
    print("SUMMARY - GraphCodeBERT, ALL RUNS PROCESSED")
    print("=" * 60)
    for lang, summary in all_results.items():
        print(f"\n--- {lang.upper()} ---")
        for col, (fixed, n_errors, rate) in summary.items():
            name = OFFICIAL_NAMES.get(col, col)
            print(f"{name:18s}: {rate:.2f}% ({fixed} fixed, {n_errors} errors)")
    print("=" * 60)


if __name__ == "__main__":
    main()