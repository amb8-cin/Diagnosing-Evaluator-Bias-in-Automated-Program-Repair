import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_dissertation_charts():
    print("Generating high-resolution charts...")
    
    # ==========================================
    # FIGURE 1: Bar Chart (Metrics)
    # ==========================================
    # Data extracted from your classification_report
    metrics = ['Precision', 'Recall', 'F1-Score']
    correct_code = [0.91, 0.90, 0.91]
    buggy_code = [0.90, 0.91, 0.91]

    x = np.arange(len(metrics))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    # Academic reading-friendly colors
    rects1 = ax1.bar(x - width/2, correct_code, width, label='Correct Code (0)', color='#4C72B0')
    rects2 = ax1.bar(x + width/2, buggy_code, width, label='Buggy (1)', color='#DD8452')

    ax1.set_ylabel('Score (0.0 to 1.0)', fontsize=12)
    ax1.set_title('Random Forest Model Performance by Class', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right', fontsize=11)

    # Add numbers on top of each bar
    ax1.bar_label(rects1, padding=3, fmt='%.2f')
    ax1.bar_label(rects2, padding=3, fmt='%.2f')

    fig1.tight_layout()
    plt.savefig('Figure_Y_Performance_Metrics.png', dpi=300, bbox_inches='tight')
    print("✅ 'Figure_Y_Performance_Metrics.png' created successfully!")

    # ==========================================
    # FIGURE 2: Confusion Matrix (Heatmap)
    # ==========================================
    # Exact data calculated from your supports (142 and 141) and recall
    confusion_matrix = np.array([[128, 14], 
                                 [12, 129]])

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    
    # Create the heatmap (blue tones usually look great in dissertations)
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                cbar=False, annot_kws={"size": 16},
                xticklabels=['Correct (0)', 'Buggy (1)'],
                yticklabels=['Correct (0)', 'Buggy (1)'])
    
    ax2.set_ylabel('True Label (Ground Truth)', fontsize=12)
    ax2.set_xlabel('Validator Prediction', fontsize=12)
    ax2.set_title('Confusion Matrix', fontsize=14, pad=15)

    # Improve axis visibility
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)

    fig2.tight_layout()
    plt.savefig('Figure_Z_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
    print("✅ 'Figure_Z_Confusion_Matrix.png' created successfully!")

if __name__ == "__main__":
    # If seaborn is not installed, run in terminal: pip install seaborn
    generate_dissertation_charts()