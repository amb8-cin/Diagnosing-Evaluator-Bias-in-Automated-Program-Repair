import argparse
import os

import joblib
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================
DEFAULT_MODEL_DIR = 'C:/Dissertacao/trained_models/'
DEFAULT_INPUT_TEMPLATE = 'C:/Dissertacao/data_bases/05_results/llm_responses_ast_{lang}.csv'  # script 15
DEFAULT_OUTPUT_TEMPLATE = 'C:/Dissertacao/data_bases/05_results/final_dissertation_report_ast_{lang}.csv'
DEFAULT_FIGURE_TEMPLATE = 'C:/Dissertacao/figuras/AST_Tournament_Scoreboard_{lang}.png'

AST_COLUMNS = {
    'Claude Opus 4.6': 'AST_Claude',
    'GPT-5.4': 'AST_GPT',
    'Gemini 3.1 Pro': 'AST_Gemini',
}


def pick(df, *names):
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"None of the columns {names} found. Available: {list(df.columns)}")


def score_one_language(lang, ast_model, ast_vectorizer, input_path, output_path, figure_path):
    if not os.path.exists(input_path):
        print(f"\n[{lang.upper()}] SKIPPED: file not found at {input_path}")
        return None

    print(f"\n[{lang.upper()}] Reading LLM AST features from: {input_path}")
    df = pd.read_csv(input_path)
    id_col = pick(df, 'Case_ID', 'ID_Caso')
    total = len(df)
    print(f"   -> Total instances analyzed: {total}")

    results = {}
    for ai_name, ast_column in AST_COLUMNS.items():
        if ast_column not in df.columns:
            print(f"   WARNING: column {ast_column} not found. Skipping {ai_name}...")
            continue

        features = df[ast_column].fillna("NO_FEATURES")

        # A response that never even compiled (SYNTAX_ERROR, from script 15) is, by definition,
        # a failed repair. We don't let the classifier guess on an out-of-vocabulary, near-zero
        # vector for these -- we score them directly as "still has bug", the same way an API
        # error is handled in scripts 16 and 18.
        syntax_error_mask = (features == "SYNTAX_ERROR")
        n_syntax_errors = int(syntax_error_mask.sum())

        verdicts = pd.Series(1, index=df.index)  # default: 1 = still has bug
        valid_idx = df.index[~syntax_error_mask]
        if len(valid_idx) > 0:
            X = ast_vectorizer.transform(features.loc[valid_idx])
            verdicts.loc[valid_idx] = ast_model.predict(X)

        df[f'verdict_{ast_column}'] = verdicts

        fixed = int((verdicts == 0).sum())
        rate = (fixed / total) * 100
        results[ai_name] = (fixed, n_syntax_errors, rate)
        print(f"   -> {ai_name}: Fixed {fixed}/{total} bugs ({rate:.2f}%), "
              f"{n_syntax_errors} responses failed to compile (SYNTAX_ERROR, counted as failures)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, sep=';')
    print(f"   Detailed report saved to '{output_path}'")

    if results:
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        ordered = dict(sorted(((k, v[2]) for k, v in results.items()), key=lambda item: item[1], reverse=True))
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
        bars = ax.bar(list(ordered.keys()), list(ordered.values()), color=colors, width=0.5)
        ax.set_ylabel('Success Rate (LFV) %', fontsize=12, fontweight='bold')
        ax.set_title(f'Final Scoreboard: LLM Repair Effectiveness (AST Validation, {lang.upper()})',
                     fontsize=13, fontweight='bold')
        ax.set_ylim(0, 100)
        for bar_item in bars:
            height = bar_item.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar_item.get_x() + bar_item.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
        plt.tight_layout()
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   Chart saved as '{figure_path}'")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['pt', 'en', 'both'], default='both')
    parser.add_argument('--input', default=None,
                         help='overrides the default path (only valid together with --lang pt or --lang en)')
    parser.add_argument('--output', default=None,
                         help='overrides the default path (only valid together with --lang pt or --lang en)')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    langs = ['pt', 'en'] if args.lang == 'both' else [args.lang]

    print("Loading the AST model and vectorizer...")
    try:
        ast_model = joblib.load(os.path.join(args.model_dir, 'validador_ast_rf.pkl'))
        ast_vectorizer = joblib.load(os.path.join(args.model_dir, 'vectorizer_ast.pkl'))
    except FileNotFoundError as e:
        print(f"ERROR: model files not found in {args.model_dir}: {e}")
        return

    all_results = {}
    for lang in langs:
        input_path = args.input or DEFAULT_INPUT_TEMPLATE.format(lang=lang)
        output_path = args.output or DEFAULT_OUTPUT_TEMPLATE.format(lang=lang)
        figure_path = DEFAULT_FIGURE_TEMPLATE.format(lang=lang)
        result = score_one_language(lang, ast_model, ast_vectorizer, input_path, output_path, figure_path)
        if result:
            all_results[lang] = result

    print("\n" + "=" * 60)
    print("SUMMARY - AST, ALL RUNS PROCESSED")
    print("=" * 60)
    for lang, results in all_results.items():
        print(f"\n--- {lang.upper()} ---")
        for ai_name, (fixed, n_syntax_errors, rate) in results.items():
            print(f"{ai_name:18s}: {rate:.2f}% ({fixed} fixed, {n_syntax_errors} syntax errors)")
    print("=" * 60)


if __name__ == "__main__":
    main()