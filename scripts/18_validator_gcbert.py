import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🤖 Starting the Semantic Evaluator (GraphCodeBERT)...")

    # ==========================================
    # 1. TRAIN THE "JUDGE" (RANDOM FOREST)
    # ==========================================
    print("📂 Loading the Judge's knowledge base...")
    embeddings_base_path = 'C:/Dissertation/data_bases/graphcodebert/embeddings_graphcodebert.csv'
    df_base = pd.read_csv(embeddings_base_path)
    
    # --- UPDATE: Removing ID_Pair to prevent string errors ---
    columns_to_remove = ['Has_Resource_Leak']
    if 'ID_Pair' in df_base.columns:
        columns_to_remove.append('ID_Pair')
        
    X_train = df_base.drop(columns=columns_to_remove).values
    y_train = df_base['Has_Resource_Leak'].values

    print(f"🌲 Training the Judge (Random Forest) with {len(X_train)} examples...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # ==========================================
    # 2. PREPARE THE NEURAL EXTRACTOR
    # ==========================================
    print("\n📥 Loading Microsoft's extractor model...")
    model_name = "microsoft/graphcodebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def extract_embedding(code):
        # If the LLM didn't respond or gave an error, we return a zero vector
        if pd.isna(code) or str(code).strip() == "" or str(code).startswith("ERROR"):
            return np.zeros(768)
            
        inputs = tokenizer(str(code), return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0, 0, :].cpu().numpy()

    # ==========================================
    # 3. EVALUATE LLMs
    # ==========================================
    tournament_path = 'C:/Dissertation/data_bases/05_results/llms_tournament_results.csv'
    print(f"\n📂 Reading tournament results: {tournament_path}")
    df_llms = pd.read_csv(tournament_path)
    
    total_cases = len(df_llms)
    results = {"GPT": 0, "Claude": 0, "Gemini": 0}

    model_columns = {'fix_gpt': 'GPT', 'fix_claude': 'Claude', 'fix_gemini': 'Gemini'}

    for df_col, llm_name in model_columns.items():
        print(f"\n🧠 Extracting semantics and evaluating {llm_name}'s responses...")
        
        # Error handling in case an LLM column does not exist in the CSV
        if df_col not in df_llms.columns:
            print(f"⚠️ Warning: Column '{df_col}' not found in CSV. Skipping {llm_name}.")
            continue
            
        generated_codes = df_llms[df_col].tolist()
        
        llm_vectors = []
        for i, code in enumerate(generated_codes):
            if i % 50 == 0 and i > 0:
                print(f"   -> Analyzed {i}/{total_cases}...")
            llm_vectors.append(extract_embedding(code))
            
        # The Judge evaluates all codes from this LLM at once
        predictions = rf.predict(llm_vectors)
        
        # Counts how many the model predicted as 0 (No Bug / Fixed)
        fixed_count = np.sum(predictions == 0)
        results[llm_name] = fixed_count

    # ==========================================
    # 4. DISPLAY FINAL RESULTS
    # ==========================================
    print("\n" + "="*60)
    print("🏆 SEMANTIC EVALUATOR RESULTS (GraphCodeBERT)")
    print("="*60)
    
    # Sorts from highest to lowest
    ranking = sorted(results.items(), key=lambda item: item[1], reverse=True)
    
    for llm_name, fixed_count in ranking:
        rate = (fixed_count / total_cases) * 100
        print(f"-> {llm_name}: Fixed {fixed_count}/{total_cases} bugs ({rate:.2f}%)")
    print("="*60)

    # ==========================================
    # 4. GENERATE THE FINAL SCORE GRAPH
    # ==========================================
    if results:
        print("\n🎨 Generating final tournament scoreboard graph...")
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # 1. Map short dictionary names to formal dissertation names
        official_names = {
            "Claude": "Claude 4.6 Opus",
            "GPT": "GPT-5.4",
            "Gemini": "Gemini 3.1 Pro"
        }
        
        # 2. Transform absolute numbers into percentages (%)
        percentage_results = {}
        for short_name, hits in results.items():
            percentage_rate = (hits / total_cases) * 100
            formal_name = official_names.get(short_name, short_name)
            percentage_results[formal_name] = percentage_rate
        
        # 3. Sort results from highest to lowest (based on percentage)
        ordered_results = dict(sorted(percentage_results.items(), key=lambda item: item[1], reverse=True))
        
        names = list(ordered_results.keys())
        rates = list(ordered_results.values())
        
        # Exact color palette (Green, Orange, Blue)
        colors = ['#2CA02C', '#FF7F0E', '#1F77B4']
        
        # Create bars without borders to match the example
        bars = ax.bar(names, rates, color=colors, width=0.5, edgecolor='none')
        
        # Update labels and titles (LFV removed, as it's now Semantic)
        ax.set_ylabel('Semantic Success Rate %', fontsize=12, fontweight='bold')
        ax.set_title('Final Score: LLM Repair Efficacy (GraphCodeBERT Validation)', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        
        # Add percentage labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        graph_filename = 'Final_Score_GraphCodeBERT.png'
        plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Graph saved successfully as '{graph_filename}'")

if __name__ == "__main__":
    main()