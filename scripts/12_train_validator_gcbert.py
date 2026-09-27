import os
import sys

# split_utils.py may live next to this script or in a "util" folder (next to it or one level up)
_HERE = os.path.dirname(os.path.abspath(__file__))
for _folder in (_HERE, os.path.join(_HERE, 'util'), os.path.join(_HERE, '..', 'util')):
    if _folder not in sys.path:
        sys.path.append(_folder)

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from split_utils import group_cv_indices, original_case, save_validator_figures

# ==========================================================
# CONFIGURATION
# ==========================================================
DEFAULT_EMBEDDINGS = 'C:/Dissertacao/data_bases/04_final/graphcodebert/embeddings_graphcodebert.csv'  # script 07
DEFAULT_FIGURES_DIR = './figuras/'
N_FOLDS = 5


def train_semantic_model(embeddings_path, figures_dir, show=False):
    print("Starting Semantic Model Training (Random Forest)...")

    print("Loading saved vectors...")
    df = pd.read_csv(embeddings_path)

    y = df['Has_Resource_Leak'].values

    # Groups: the ORIGINAL CASE (all synthetic siblings of a case stay on the same side)
    groups = df['Pair_ID'].apply(original_case).values
    print(f"Examples: {len(df)} | original cases (groups): {len(set(groups))}")

    # X only contains the numeric columns dim_0, dim_1, ...
    X = df.drop(columns=['Has_Resource_Leak', 'Pair_ID']).values

    print(f"\nInternal evaluation: {N_FOLDS}-fold cross-validation by original case...")
    oof_pred = np.zeros(len(df), dtype=int)
    fold_acc = []
    for k, (train_idx, test_idx) in enumerate(group_cv_indices(groups, N_FOLDS), 1):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X[train_idx], y[train_idx])
        pred = rf.predict(X[test_idx])
        oof_pred[test_idx] = pred
        fold_acc.append(accuracy_score(y[test_idx], pred))
        print(f"   fold {k}: {len(set(groups[test_idx]))} cases, "
              f"{len(test_idx)} examples, accuracy {fold_acc[-1] * 100:.2f}%")

    print("\nFINAL RESULTS (GraphCodeBERT, split by original case):")
    print("=" * 60)
    print(f"Accuracy per fold: mean {np.mean(fold_acc) * 100:.2f}% (std {np.std(fold_acc) * 100:.2f} pp)")
    print(f"Pooled accuracy (all test folds together): {accuracy_score(y, oof_pred) * 100:.2f}%")
    print("-" * 60)
    print("Confusion Matrix (pooled over the test folds):")
    print(confusion_matrix(y, oof_pred))
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y, oof_pred))
    print("=" * 60)
    save_validator_figures(
        y, oof_pred, figures_dir,
        metrics_file='Figura_GCBERT_Metricas_Desempenho.png',
        matrix_file='Figura_GCBERT_Matriz_Confusao.png',
        colors=('#4C72B0', '#DD8452'), cmap='Blues',
        metrics_title='Métricas de Avaliação do Validador Semântico GraphCodeBERT',
        matrix_title='Matriz de Confusão: Validador Semântico GraphCodeBERT',
        show=show)
    print("Note: script 18 retrains the Random Forest on 100% of these embeddings for the tournament.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', default=DEFAULT_EMBEDDINGS)
    parser.add_argument('--figures-dir', default=DEFAULT_FIGURES_DIR)
    parser.add_argument('--show', action='store_true', help='also open the figures on screen')
    a = parser.parse_args()
    train_semantic_model(a.embeddings, a.figures_dir, a.show)