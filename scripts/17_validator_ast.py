import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🏆 Starting the LLM Tournament (AST Validator)...\n")

    # 1. LOAD ORACLE AND VECTORIZER
    print("1. Loading the AST model and vectorizer...")
    try:
        ast_model = joblib.load('C:/Dissertacao/trained_models/validador_ast_rf.pkl')
        ast_vectorizer = joblib.load('C:/Dissertacao/trained_models/vectorizer_ast.pkl')
    except FileNotFoundError:
        print("❌ ERROR: Model files not found. Check the path.")
        return

    # 2. LOAD LLM DATA
    # Replace with the actual path to your CSV containing LLM responses/features
    responses_path = 'C:/Dissertacao/data_bases/ast/answers_llms_ast.csv' 
    print(f"2. Reading LLM responses from: {responses_path}...")
    
    try:
        df_llms = pd.read_csv(responses_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File {responses_path} not found.")
        return

    total_bugs = len(df_llms)
    print(f"   -> Total instances analyzed: {total_bugs}\n")

    # 3. VECTORIZE AND EVALUATE EACH MODEL
    # IMPORTANT: Put here the exact names of the columns containing the AST features for each AI
    ai_models = {
        'Claude 4.6 Opus': 'AST_Claude',
        'GPT-5.4': 'AST_GPT',
        'Gemini 3.1 Pro': 'AST_Gemini'
    }

    results = {}

    print("3. Processing Oracle predictions...")
    for ai_name, ast_column in ai_models.items():
        if ast_column not in df_llms.columns:
            print(f"   ⚠️ Column {ast_column} not found in CSV. Skipping {ai_name}...")
            continue
            
        # Handle nulls in case any AI did not generate valid code
        ai_features = df_llms[ast_column].fillna("NO_FEATURES")
        
        # Vectorize using the 9-node vocabulary learned during training
        X_ai_vectorized = ast_vectorizer.transform(ai_features)
        
        # Predict (0 = Fixed, 1 = Bug)
        predictions = ast_model.predict(X_ai_vectorized)
        
        # Count how many zeros (fixed codes) the AI achieved
        successes = (predictions == 0).sum()
        success_rate = (successes / total_bugs) * 100
        
        results[ai_name] = success_rate
        print(f"   -> {ai_name}: Fixed {successes}/{total_bugs} bugs ({success_rate:.2f}%)")

    # 4. GENERATE FINAL SCOREBOARD CHART
    if results:
        print("\n🎨 Generating final tournament scoreboard chart...")
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Sort results from highest to lowest
        ordered_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        
        names = list(ordered_results.keys())
        rates = list(ordered_results.values())
        
        # Modern color palette
        colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
        
        bars = ax.bar(names, rates, color=colors, width=0.5)
        
        ax.set_ylabel('Success Rate (LFV) %', fontsize=12, fontweight='bold')
        ax.set_title('Final Scoreboard: LLM Repair Effectiveness (AST Validation)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Add percentage labels on top of the bars
        for bar_item in bars:
            height = bar_item.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar_item.get_x() + bar_item.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('AST_Tournament_Scoreboard_Figure.png', dpi=300)
        print("✅ Chart saved as 'AST_Tournament_Scoreboard_Figure.png'")
        plt.show()

if __name__ == "__main__":
    main()