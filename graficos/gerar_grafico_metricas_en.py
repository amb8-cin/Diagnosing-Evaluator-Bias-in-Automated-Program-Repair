import matplotlib.pyplot as plt
import numpy as np

def generate_metrics_chart():
    print("📊 Generating the metrics bar chart (Figure Y)...")
    
    # Metric names
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    # UPDATED VALUES TO MATCH DISSERTATION TEXT (90.81% Model)
    class_0_values = [0.91, 0.90, 0.91] # Secure Code
    class_1_values = [0.90, 0.91, 0.91] # Buggy Code
    
    # Bar position configuration
    x = np.arange(len(metrics))
    width = 0.35 # Width of each bar
    
    # Set chart style and size
    plt.style.use('seaborn-v0_8-whitegrid') # White background with soft grid lines
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw bars (using sober academic colors: Blue and Dark Orange)
    rects1 = ax.bar(x - width/2, class_0_values, width, label='Secure Code (0)', color='#2ca02c')
    rects2 = ax.bar(x + width/2, class_1_values, width, label='Buggy Code (1)', color='#d62728')
    
    # Add titles and labels
    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, weight='bold')
    ax.set_title('Evaluation Metrics: Random Forest Classifier', fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, weight='bold')
    
    # Limit Y axis from 0 to 1.1 to prevent text from sticking to the top
    ax.set_ylim(0, 1.1)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    
    # Add exact numbers on top of each bar
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=11)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=11)
    
    # Adjust layout and save high-resolution image (300 dpi)
    plt.tight_layout()
    file_name = 'Figure_Y_Performance_Metrics_en.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    print(f"✅ Success! High-resolution chart saved as: {file_name}")

if __name__ == "__main__":
    generate_metrics_chart()