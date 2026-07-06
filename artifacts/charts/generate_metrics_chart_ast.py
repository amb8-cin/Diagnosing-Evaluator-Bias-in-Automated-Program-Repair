import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Elegant style configuration for academic papers
sns.set_theme(style="whitegrid")

def plot_ast_metrics():
    # Data extracted from your AST training log
    metrics = ['Precision', 'Recall', 'F1-Score']
    fixed_code = [0.91, 0.87, 0.89]
    buggy_code = [0.87, 0.91, 0.89]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colors: Green for Fixed, Red for Bug (makes it easier for the reviewing committee to read)
    rects1 = ax.bar(x - width/2, fixed_code, width, label='Fixed (0)', color='#2ca02c')
    rects2 = ax.bar(x + width/2, buggy_code, width, label='Buggy (1)', color='#d62728')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance of the Structural Validator (AST)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1) # Extends to 1.1 to leave room for the legend
    ax.legend(loc='lower right', fontsize=11)

    # Adds the numbers on top of each bar
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=11)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=11)

    plt.tight_layout()
    plt.savefig('Figure_Y_AST_Performance_Metrics.png', dpi=300, bbox_inches='tight')
    print("✅ Metrics chart saved as 'Figure_Y_AST_Performance_Metrics.png'")

def plot_ast_confusion_matrix():
    # Actual Confusion Matrix values from your log
    # [[TN, FP], [FN, TP]] -> [[117, 18], [12, 123]]
    matrix = np.array([[117, 18], [12, 123]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Using green shades ('Greens') to visually differentiate from the TF-IDF which was blue
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Fixed (0)', 'Buggy (1)'],
                yticklabels=['Fixed (0)', 'Buggy (1)'],
                annot_kws={"size": 16, "fontweight": "bold"})
    
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Oracle Prediction (AST)', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - AST Model', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Figure_Z_AST_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion matrix saved as 'Figure_Z_AST_Confusion_Matrix.png'")

if __name__ == "__main__":
    print("📊 Generating high-resolution images for LaTeX...")
    plot_ast_metrics()
    plot_ast_confusion_matrix()
    print("🎉 Charts successfully generated in your folder!")