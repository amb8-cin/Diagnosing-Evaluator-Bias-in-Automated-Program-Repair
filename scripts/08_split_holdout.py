import argparse
import os
import re

import numpy as np
import pandas as pd

# ==========================================================
# CONFIGURATION
# ==========================================================
DEFAULT_INPUT = './data_bases/04_final/dataset_synthetic_chatgpt.csv'
DEFAULT_OUTPUT_DIR = './data_bases/04_final/'
TARGET_PAIRS = 300   # approximate size of the LLM tournament holdout (in bug/fix pairs)
SEED = 42            # keeps the selection reproducible

# Accepted names for each column (Portuguese and/or English, even mixed in the same file)
COLUMN_CANDIDATES = {
    'id': ['Case_ID', 'ID_Caso'],
    'code': ['Code_Snippet', 'Codigo_Snippet'],
    'label': ['Has_Resource_Leak', 'Has_Resource_Leak'],
}
# Name of the bug-code column in the file given to the LLMs (script 13 reads this name)
DEFAULT_BUG_CODE_COLUMN = 'code_with_bug'

SYNTHETIC_SUFFIX = re.compile(r'_(?:SINTETICO|SYNTHETIC)_\d+(?:_(?:BUG|FIX))?$')
BUG_FIX_SUFFIX = re.compile(r'_(?:BUG|FIX)$')


def original_case(case_id):
    """
    Returns the ORIGINAL DroidLeaks case a row came from.
    Ex: 'Case_003_AnkiDroid_SINTETICO_3_BUG' -> 'Case_003_AnkiDroid'
    All synthetic siblings of the same original case share this value.
    """
    case_id = str(case_id)
    stripped = SYNTHETIC_SUFFIX.sub('', case_id)
    if stripped == case_id:  # not synthetic: only remove _BUG / _FIX
        stripped = BUG_FIX_SUFFIX.sub('', case_id)
    return stripped


def pair_id(case_id):
    """Ex: 'Case_003_AnkiDroid_SINTETICO_3_BUG' -> 'Case_003_AnkiDroid_SINTETICO_3'"""
    return str(case_id).rsplit('_', 1)[0]


def detect_columns(df):
    cols = {}
    for role, names in COLUMN_CANDIDATES.items():
        found = [n for n in names if n in df.columns]
        if not found:
            raise ValueError(f"No column for '{role}' found (looked for {names}). "
                             f"Columns in the file: {list(df.columns)}")
        cols[role] = found[0]
    return cols


def choose_holdout_cases(cases_pairs, target, seed):
    """
    Randomly picks ORIGINAL CASES (never loose pairs) until the number of pairs
    is as close as possible to the target. All siblings of a chosen case go together,
    so the holdout size may differ slightly from the target.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(cases_pairs.index.to_numpy())
    chosen, total = [], 0
    for case in order:
        n = int(cases_pairs[case])
        if total + n > target:
            # add this last case only if it leaves us closer to the target
            if abs(total + n - target) < abs(total - target):
                chosen.append(case)
                total += n
            break
        chosen.append(case)
        total += n
    return chosen, total


def original_app(case):
    """Ex: 'Case_003_AnkiDroid' -> 'AnkiDroid'"""
    return re.sub(r'^Case_\d+_', '', case)


def choose_holdout_cases_stratified(cases_pairs, target, seed, tries=5000):
    """
    Picks ORIGINAL CASES stratified by original app: every app with 2+ cases gets about
    the same share of its cases in the holdout (at least 1, never all), so every app is
    present on both sides. Apps with a single case stay in training.
    Many random draws are tried (deterministic for a given seed) and the one whose number
    of pairs is closest to the target is kept.
    """
    fraction = target / cases_pairs.sum()
    by_app = {}
    for case in cases_pairs.index:
        by_app.setdefault(original_app(case), []).append(case)

    quota = {}
    for app, cases in by_app.items():
        k = len(cases)
        quota[app] = 0 if k < 2 else min(k - 1, max(1, int(round(k * fraction))))

    rng = np.random.default_rng(seed)
    best, best_total = None, None
    for _ in range(tries):
        chosen = []
        for app, cases in by_app.items():
            if quota[app] > 0:
                chosen += list(rng.choice(cases, size=quota[app], replace=False))
        total = int(cases_pairs[chosen].sum())
        if best is None or abs(total - target) < abs(best_total - target):
            best, best_total = chosen, total
            if best_total == target:
                break
    return best, best_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--target-pairs', type=int, default=TARGET_PAIRS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--bug-code-column', default=DEFAULT_BUG_CODE_COLUMN,
                        help='name of the bug-code column in the file for the LLMs')
    parser.add_argument('--no-stratify', action='store_true',
                        help='plain random choice of cases, without balancing by app')
    args = parser.parse_args()

    # 1. Load your dataset
    df = pd.read_csv(args.input, sep=';')
    cols = detect_columns(df)
    id_col, code_col, label_col = cols['id'], cols['code'], cols['label']

    # 2. Create the Pair ID and the ORIGINAL CASE ID
    df['Pair_ID'] = df[id_col].apply(pair_id)
    df['Original_Case'] = df[id_col].apply(original_case)

    n_pairs_total = df['Pair_ID'].nunique()
    cases_pairs = df.groupby('Original_Case')['Pair_ID'].nunique()
    print(f"Total pairs (bug/fix): {n_pairs_total}")
    print(f"Total original cases: {len(cases_pairs)} "
          f"(pairs per case: min {cases_pairs.min()}, max {cases_pairs.max()})")

    # 3. Select ORIGINAL CASES for the tournament (all siblings stay together)
    if args.no_stratify:
        holdout_cases, n_holdout_pairs = choose_holdout_cases(cases_pairs, args.target_pairs, args.seed)
    else:
        holdout_cases, n_holdout_pairs = choose_holdout_cases_stratified(
            cases_pairs, args.target_pairs, args.seed)
    holdout_mask = df['Original_Case'].isin(holdout_cases)

    # 4. Dataset for the LLM tournament: only the BUG version of the held-out cases
    df_llm = df[holdout_mask & (df[label_col] == 1)].copy()
    df_llm = df_llm.rename(columns={code_col: args.bug_code_column})

    # 5. Dataset for validator training: everything from the OTHER original cases
    df_train = df[~holdout_mask].copy()

    # 6. Full held-out pairs (BUG + FIX), useful to test the validators later
    df_holdout_full = df[holdout_mask].copy()

    # 7. Safety checks
    assert set(df_train['Original_Case']).isdisjoint(set(df_llm['Original_Case'])), \
        "Leakage: an original case appears in both training and holdout!"
    assert df_llm['Pair_ID'].is_unique, "Duplicated bug in the holdout!"
    bug_fix_counts = df_holdout_full.groupby('Pair_ID')[label_col].nunique()
    assert (bug_fix_counts == 2).all(), "Some held-out pair is missing its BUG or FIX row!"

    # 8. Remove auxiliary columns and save
    aux = ['Pair_ID', 'Original_Case']
    os.makedirs(args.output_dir, exist_ok=True)
    df_llm.drop(columns=aux).to_csv(
        os.path.join(args.output_dir, 'holdout_300_bugs_llm.csv'), index=False)
    df_train.drop(columns=aux).to_csv(
        os.path.join(args.output_dir, 'train_validator_final.csv'), index=False, sep=';')
    df_holdout_full.drop(columns=aux).to_csv(
        os.path.join(args.output_dir, 'holdout_pairs_bug_and_fix.csv'), index=False, sep=';')
    with open(os.path.join(args.output_dir, 'holdout_original_cases.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(holdout_cases)))

    # 9. Report
    print("\n--- Division Report ---")
    print(f"Held-out original cases: {len(holdout_cases)} of {len(cases_pairs)}")
    print(f"File for LLMs: {len(df_llm)} bug rows ({n_holdout_pairs} pairs; target was {args.target_pairs})")
    print(f"File for Validator: {len(df_train)} rows ({df_train['Pair_ID'].nunique()} pairs, "
          f"{df_train['Original_Case'].nunique()} original cases)")
    print("Leakage check passed: no original case is shared between training and holdout.")

    # Distribution by ORIGINAL app (taken from the case id, e.g. Case_003_AnkiDroid -> AnkiDroid)
    held = df_llm['Original_Case'].map(original_app).value_counts()
    total = df[df[label_col] == 1]['Original_Case'].map(original_app).value_counts()
    print("\nBugs per original app (holdout / total):")
    for app in total.index:
        print(f"  {app}: {held.get(app, 0)} / {total[app]}")


if __name__ == "__main__":
    main()