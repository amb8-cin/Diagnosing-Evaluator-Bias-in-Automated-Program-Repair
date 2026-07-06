import pandas as pd
import os

def align_ast_split(full_ast_path, real_train_path, llm_holdout_path, output_folder):
    print("🔄 Aligning the AST Split with official Tournament data...")
    
    # 1. Load files
    df_ast = pd.read_csv(full_ast_path)
    df_official_train = pd.read_csv(real_train_path, sep=';')
    df_official_holdout = pd.read_csv(llm_holdout_path) # the 300 LLM bugs
    
    # 2. Create Pair ID in AST for mapping
    df_ast['Pair_ID'] = df_ast['Case_ID'].apply(lambda x: str(x).rsplit('_', 1)[0])
    
    # Create Pair ID in official files to serve as key
    df_official_train['Pair_ID'] = df_official_train['Case_ID'].apply(lambda x: str(x).rsplit('_', 1)[0])
    df_official_holdout['Pair_ID'] = df_official_holdout['Case_ID'].apply(lambda x: str(x).replace('_BUG', '').replace('_FIX', ''))
    
    train_pair_list = df_official_train['Pair_ID'].unique()
    holdout_pair_list = df_official_holdout['Pair_ID'].unique()
    
    # 3. Separate AST using official lists (Ensures parity and prevents Data Leakage)
    df_ast_train = df_ast[df_ast['Pair_ID'].isin(train_pair_list)].drop(columns=['Pair_ID'])
    df_ast_test = df_ast[df_ast['Pair_ID'].isin(holdout_pair_list)].drop(columns=['Pair_ID'])
    
    # 4. Create AST holdout (Only bugs from the test group)
    df_ast_holdout_bugs = df_ast_test[df_ast_test['Has_Resource_Leak'] == 1]
    
    # Save the perfectly aligned files
    os.makedirs(output_folder, exist_ok=True)
    df_ast_train.to_csv(os.path.join(output_folder, "dataset_ast_train.csv"), index=False)
    df_ast_test.to_csv(os.path.join(output_folder, "dataset_ast_test.csv"), index=False)
    df_ast_holdout_bugs.to_csv(os.path.join(output_folder, "holdout_ast.csv"), index=False)
    
    print("\n✅ Alignment complete! Your models are now scientifically paired.")
    print(f"📈 Train AST: {len(df_ast_train)} rows | 🎯 Test/Holdout AST: {len(df_ast_holdout_bugs)} bugs.")

if __name__ == "__main__":
    FULL_AST = "C:/Dissertacao/data_bases/04_final/dataset_ast_full.csv"
    REAL_TRAIN = "C:/Dissertacao/data_bases/04_final/train_validator_final.csv"
    LLM_HOLDOUT = "C:/Dissertacao/data_bases/04_final/holdout_300_bugs_llm.csv"
    DESTINATION_FOLDER = "C:/Dissertacao/data_bases/04_final/"
    
    align_ast_split(FULL_AST, REAL_TRAIN, LLM_HOLDOUT, DESTINATION_FOLDER)