"""
Shared helpers for the internal evaluation of the three validators (scripts 10, 11 and 12).

Why this file exists:
each original DroidLeaks case generated ~10 synthetic pairs ("siblings") that are almost
identical to each other. If siblings end up on both sides of a train/test split, the test is
not really "unseen" and the accuracy is inflated. Here the split is done by ORIGINAL CASE
(ex: Case_003_AnkiDroid), and the same fold assignment is used by the three scripts, so the
three validators are evaluated on exactly the same held-out cases and are comparable.

Put this file in the same folder as scripts 10, 11 and 12.
"""
import re

import numpy as np

SYNTHETIC_SUFFIX = re.compile(r'_(?:SINTETICO|SYNTHETIC)_\d+(?:_(?:BUG|FIX))?$')
BUG_FIX_SUFFIX = re.compile(r'_(?:BUG|FIX)$')


def original_case(case_id):
    """
    'Case_003_AnkiDroid_SINTETICO_3_BUG' -> 'Case_003_AnkiDroid'
    'Case_003_AnkiDroid_SINTETICO_3'     -> 'Case_003_AnkiDroid'  (a Pair_ID also works)
    """
    case_id = str(case_id)
    stripped = SYNTHETIC_SUFFIX.sub('', case_id)
    if stripped == case_id:  # not synthetic: only remove _BUG / _FIX
        stripped = BUG_FIX_SUFFIX.sub('', case_id)
    return stripped


def group_cv_indices(groups, n_splits=5, seed=42):
    """
    Cross-validation by group: every original case goes entirely to ONE fold.
    Deterministic: the same set of cases always gives the same folds, in any script.
    Yields (train_indices, test_indices) for each fold.
    """
    groups = np.asarray(groups)
    cases = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(cases))
    fold_of = {cases[idx]: pos % n_splits for pos, idx in enumerate(order)}
    folds = np.array([fold_of[g] for g in groups])
    for k in range(n_splits):
        yield np.where(folds != k)[0], np.where(folds == k)[0]


def save_validator_figures(y_true, y_pred, out_dir, metrics_file, matrix_file, colors,
                           cmap, metrics_title, matrix_title, show=False):
    """
    Saves the two figures used in the dissertation (PNG, 300 dpi):
      1) bar chart with Precision / Recall / F1-Score for both classes
      2) confusion matrix with the counts
    Both are computed from the pooled cross-validation predictions.
    """
    import os
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_names = ['Código Seguro (0)', 'Com Bug (1)']

    # ---- Figure 1: Precision / Recall / F1 ----
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    groups = ['Precision\n(Precisão)', 'Recall\n(Sensibilidade)', 'F1-Score\n(Equilíbrio)']
    values = [(precision[0], precision[1]), (recall[0], recall[1]), (f1[0], f1[1])]
    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    bars0 = ax.bar(x - width / 2, [v[0] for v in values], width, label=class_names[0], color=colors[0])
    bars1 = ax.bar(x + width / 2, [v[1] for v in values], width, label=class_names[1], color=colors[1])
    for bars in (bars0, bars1):
        for bar in bars:
            ax.annotate(f'{bar.get_height():.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Pontuação (0.0 a 1.0)', fontsize=11, fontweight='bold')
    ax.set_title(metrics_title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right')
    fig.tight_layout()
    path1 = os.path.join(out_dir, metrics_file)
    fig.savefig(path1, dpi=300, bbox_inches='tight')

    # ---- Figure 2: Confusion matrix ----
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.imshow(matrix, cmap=cmap)
    threshold = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f'{matrix[i, j]}', ha='center', va='center', fontsize=16, fontweight='bold',
                     color='white' if matrix[i, j] > threshold else 'black')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Previsão do Modelo', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Rótulo Verdadeiro (Ground Truth)', fontsize=11, fontweight='bold')
    ax2.set_title(matrix_title, fontsize=13, fontweight='bold', pad=12)
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Número de Casos')
    fig2.tight_layout()
    path2 = os.path.join(out_dir, matrix_file)
    fig2.savefig(path2, dpi=300, bbox_inches='tight')

    print(f"\nFigures saved:\n   {path1}\n   {path2}")
    if show:
        plt.show()
    plt.close('all')