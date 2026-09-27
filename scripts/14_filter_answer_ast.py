import argparse
import os

import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================
DEFAULT_AST_IDS = 'C:/Dissertacao/data_bases/04_final/holdout_ast.csv'  # output of script 9 (~295 bugs)
DEFAULT_INPUT_TEMPLATE = 'C:/Dissertacao/data_bases/05_results/tournament_results_holdout_v2_{lang}.csv'
DEFAULT_OUTPUT_TEMPLATE = 'C:/Dissertacao/data_bases/05_results/tournament_results_FILTERED_AST_{lang}.csv'


def pick(df, *names):
    for name in names:
        if name in df.columns:
            return name
    raise ValueError(f"None of the columns {names} found. Available: {list(df.columns)}")


def filter_one_language(lang, df_ast_ids, input_path, output_path):
    if not os.path.exists(input_path):
        print(f"[{lang.upper()}] SKIPPED: file not found at {input_path}")
        return None

    df_responses = pd.read_csv(input_path)
    resp_id_col = pick(df_responses, 'Case_ID', 'ID_Caso')

    # Inner join: keeps only the bugs that also survived the AST parsing during training alignment
    # (script 9), so TF-IDF, AST and GraphCodeBERT can be compared on the exact same case set.
    df_filtered = df_ast_ids.merge(
        df_responses, left_on='_ast_id', right_on=resp_id_col, how='inner'
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_csv(output_path, index=False, encoding='utf-8')

    print(f"[{lang.upper()}] Filtered instances: {len(df_filtered)} (parity with the AST-eligible holdout). "
          f"Saved to {output_path}")
    return len(df_filtered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['pt', 'en', 'both'], default='both')
    parser.add_argument('--ast-ids', default=DEFAULT_AST_IDS)
    parser.add_argument('--input', default=None,
                         help='overrides the default path (only valid together with --lang pt or --lang en)')
    parser.add_argument('--output', default=None,
                         help='overrides the default path (only valid together with --lang pt or --lang en)')
    args = parser.parse_args()

    langs = ['pt', 'en'] if args.lang == 'both' else [args.lang]

    print("Filtering LLM responses for the AST universe...")
    df_ast = pd.read_csv(args.ast_ids)
    ast_id_col = pick(df_ast, 'Case_ID', 'ID_Caso')
    df_ast_ids = df_ast[[ast_id_col]].rename(columns={ast_id_col: '_ast_id'})

    for lang in langs:
        input_path = args.input or DEFAULT_INPUT_TEMPLATE.format(lang=lang)
        output_path = args.output or DEFAULT_OUTPUT_TEMPLATE.format(lang=lang)
        filter_one_language(lang, df_ast_ids, input_path, output_path)

    print("\nDone. Next step: script 15 (convert these corrections to AST features).")


if __name__ == "__main__":
    main()