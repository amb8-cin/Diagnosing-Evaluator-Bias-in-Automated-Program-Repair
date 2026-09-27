import argparse
import os
import re

import joblib
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================
DEFAULT_MODEL_DIR = 'C:/Dissertacao/trained_models/'
DEFAULT_INPUT_TEMPLATE = 'C:/Dissertacao/data_bases/05_results/tournament_results_holdout_v2_{lang}.csv'
DEFAULT_OUTPUT_TEMPLATE = 'C:/Dissertacao/data_bases/05_results/final_dissertation_report_tfidf_{lang}.csv'

LLM_COLUMNS = ['fix_gpt', 'fix_claude', 'fix_gemini']


def pick(df, *names):
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"None of the columns {names} found. Available: {list(df.columns)}")


# Must stay identical to the cleaning used when training the validator (script 10).
def clean_java_code(code):
    if not isinstance(code, str):
        return ""
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code)
    return code.strip().lower()


def is_error(value):
    return not (pd.notna(value) and str(value).strip() != "" and not str(value).strip().startswith("ERROR"))


def score_one_language(lang, model, vectorizer, input_path, output_path):
    if not os.path.exists(input_path):
        print(f"\n[{lang.upper()}] SKIPPED: file not found at {input_path}")
        return None

    print(f"\n[{lang.upper()}] Loading tournament results from {input_path}...")
    df = pd.read_csv(input_path)
    pick(df, 'Case_ID', 'ID_Caso')  # just to fail fast if the file is malformed
    total = len(df)
    print(f"   -> {total} bugs loaded")

    summary = {}
    for llm in LLM_COLUMNS:
        if llm not in df.columns:
            print(f"   WARNING: column '{llm}' not found, skipping.")
            continue

        error_mask = df[llm].apply(is_error)
        n_errors = int(error_mask.sum())

        verdicts = pd.Series(1, index=df.index)  # default: 1 = still has bug (also used for API errors)
        valid_idx = df.index[~error_mask]
        if len(valid_idx) > 0:
            cleaned = df.loc[valid_idx, llm].apply(clean_java_code)
            X = vectorizer.transform(cleaned)
            verdicts.loc[valid_idx] = model.predict(X)

        df[f'verdict_{llm}'] = verdicts  # 0 = safe/fixed, 1 = still has bug (or API error)

        fixed = int((verdicts == 0).sum())
        rate = (fixed / total) * 100
        summary[llm] = (fixed, n_errors, rate)

    print(f"   FINAL TOURNAMENT SCORE - TF-IDF ({lang.upper()})")
    print("   " + "-" * 46)
    for llm, (fixed, n_errors, rate) in summary.items():
        print(f"   {llm.upper():12s}: {fixed}/{total} fixed ({rate:.2f}%), {n_errors} API errors counted as failures")

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
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    langs = ['pt', 'en'] if args.lang == 'both' else [args.lang]

    print("Loading the Judge (TF-IDF Random Forest)...")
    model = joblib.load(os.path.join(args.model_dir, 'resource_leak_validator.pkl'))
    vectorizer = joblib.load(os.path.join(args.model_dir, 'vectorizer.pkl'))

    all_results = {}
    for lang in langs:
        input_path = args.input or DEFAULT_INPUT_TEMPLATE.format(lang=lang)
        output_path = args.output or DEFAULT_OUTPUT_TEMPLATE.format(lang=lang)
        result = score_one_language(lang, model, vectorizer, input_path, output_path)
        if result:
            all_results[lang] = result

    print("\n" + "=" * 60)
    print("SUMMARY - TF-IDF, ALL RUNS PROCESSED")
    print("=" * 60)
    for lang, summary in all_results.items():
        print(f"\n--- {lang.upper()} ---")
        for llm, (fixed, n_errors, rate) in summary.items():
            print(f"{llm.upper():12s}: {rate:.2f}% ({fixed} fixed, {n_errors} errors)")
    print("=" * 60)


if __name__ == "__main__":
    main()