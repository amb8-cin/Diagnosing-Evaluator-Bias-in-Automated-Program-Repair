import matplotlib.pyplot as plt
import numpy as np

def generate_graphcodebert_metrics_chart():
    print("📊 Generating the metrics chart (GraphCodeBERT)...")
    
    # Metric names
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    # ACTUAL VALUES FROM YOUR CLASSIFICATION REPORT
    # Class 0: 0.84, 0.80, 0.82
    # Class 1: 0.81, 0.85, 0.83
    class_0_values = [0.84, 0.80, 0.82] 
    class_1_values = [0.81, 0.85, 0.83] 
    
    x = np.arange(len(metrics))
    width = 0.35 
    
    # White style with soft grid
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Academic colors (Sober Blue and Orange)
    rects1 = ax.bar(x - width/2, class_0_values, width, label='Secure Code (0)', color='#4C72B0')
    rects2 = ax.bar(x + width/2, class_1_values, width, label='Buggy Code (1)', color='#DD8452')
    
    # Label configurations
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, weight='bold')
    ax.set_title('Evaluation Metrics of the GraphCodeBERT Semantic Validator', fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, weight='bold')
    
    ax.set_ylim(0, 1.0) # Correct scale from 0 to 1
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    
    # Add values on top of the bars
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=11, weight='bold')
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=11, weight='bold')
    
    plt.tight_layout()
    
    # Save in 300 DPI
    file_name = 'GraphCodeBERT_Evaluation_Metrics.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    print(f"✅ Success! Chart saved as: {file_name}")

if __name__ == "__main__":
    generate_graphcodebert_metrics_chart()