import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_graphcodebert_confusion_matrix():
    print("🎨 Generating the Confusion Matrix (GraphCodeBERT) for the dissertation...")
    
    # ACTUAL NUMBERS FROM YOUR RESULT (Accuracy 82.39%)
    # [[TN, FP], [FN, TP]]
    confusion_matrix = np.array([[170, 43], 
                                 [32, 181]])
    
    # Class names
    labels = ["Secure Code (0)", "Buggy Code (1)"]
    
    # Academic style configuration
    sns.set_theme(style="white")
    plt.figure(figsize=(8, 6))
    
    # Create the Heatmap
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=labels, yticklabels=labels,
                     annot_kws={"size": 16, "weight": "bold"},
                     cbar_kws={'label': 'Number of Cases'})
    
    # Titles and Axes
    plt.title('Confusion Matrix: GraphCodeBERT Semantic Validator', fontsize=15, pad=20, weight='bold')
    plt.ylabel('True Label (Ground Truth)', fontsize=12, weight='bold')
    plt.xlabel('Model Prediction (Accuracy 82.39%)', fontsize=12, weight='bold')
    
    plt.tight_layout()
    
    # Save in 300 DPI
    file_name = 'GraphCodeBERT_Confusion_Matrix.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    print(f"✅ Success! Chart saved as: {file_name}")

if __name__ == "__main__":
    generate_graphcodebert_confusion_matrix()