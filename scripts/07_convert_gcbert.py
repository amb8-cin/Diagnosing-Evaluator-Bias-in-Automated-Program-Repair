import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 1. We move the function outside so Python always finds it
def extract_embedding(code, tokenizer, model, device):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # We take the CLS token (global representation of the code)
    return outputs.last_hidden_state[0, 0, :].cpu().numpy()

def extract_save_embeddings():
    print("🚀 Starting Embeddings Extraction (GraphCodeBERT)...")

    model_name = "microsoft/graphcodebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 2. IMPORTANT: Using the filtered training file
    data_path = 'C:/Dissertation/data_bases/train_validator_final.csv' 
    
    try:
        df = pd.read_csv(data_path, sep=';')
    except:
        df = pd.read_csv(data_path)

    # Creating the Pair_ID to prevent Data Leakage in the trainer later
    if 'Case_ID' in df.columns:
        df['Pair_ID'] = df['Case_ID'].apply(lambda x: x.rsplit('_', 1)[0])

    # Ensuring the code column name is correct
    code_column = 'Code_Snippet' if 'Code_Snippet' in df.columns else 'code_with_bug'
    
    df = df.dropna(subset=[code_column, 'Has_Resource_Leak'])
    codes = df[code_column].astype(str).tolist()
    labels = df['Has_Resource_Leak'].astype(int).tolist()
    pair_ids = df['Pair_ID'].tolist()

    print(f"📊 Processing {len(codes)} codes...")

    X_embeddings = []
    for i, code in enumerate(codes):
        if i % 100 == 0 and i > 0:
            print(f"   -> Extracted {i}/{len(codes)}...")
        
        # Calling the function that is now above
        embedding = extract_embedding(code, tokenizer, model, device)
        X_embeddings.append(embedding)
        
    print("💾 Saving neural features and IDs to CSV...")
    
    df_embeddings = pd.DataFrame(X_embeddings)
    df_embeddings.columns = [f"dim_{i}" for i in range(768)]
    df_embeddings['Has_Resource_Leak'] = labels
    df_embeddings['Pair_ID'] = pair_ids # Saves the ID for GroupShuffleSplit
    
    output_path = 'C:/Dissertation/data_bases/graphcodebert/embeddings_graphcodebert.csv'
    df_embeddings.to_csv(output_path, index=False)
    print(f"✅ File saved successfully at: {output_path}")

if __name__ == "__main__":
    extract_save_embeddings()