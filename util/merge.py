import pandas as pd

def merge_full_ast_results():
    print("🔄 Starting triple data merge...")

    # 1. File paths
    ast_path = 'C:/Dissertation/data_bases/04_final/holdout_ast.csv' # THE MASTER LIST (290 IDs)
    raw_path = 'C:/Dissertation/data_bases/04_final/synthetic_chatgpt_dataset.csv' # THE ORIGINAL CONTEXT
    llms_path = 'C:/Dissertation/data_bases/05_results/llms_tournament_results.csv' # THE OLD ANSWERS
    
    # 2. Data reading
    try:
        df_ast = pd.read_csv(ast_path)
        df_llms = pd.read_csv(llms_path)
    except FileNotFoundError as e:
        print(f"❌ Error reading file: {e}")
        return

    # Robust reading of the raw dataset (in case it has semicolons)
    try:
        df_raw = pd.read_csv(raw_path)
    except pd.errors.ParserError:
        df_raw = pd.read_csv(raw_path, sep=';')

    # 3. STEP A: Bring the original context (Application, Resource, Code) for the 290 IDs
    # The raw file uses 'Code_Snippet', let's rename it to 'buggy_code' to maintain consistency
    df_raw = df_raw.rename(columns={'Codigo_Snippet': 'buggy_code'})
    
    # Perform a Left Join with the raw dataset
    df_base_context = pd.merge(
        df_ast[['ID_Caso']], 
        df_raw[['ID_Caso', 'Application', 'Resource_Class', 'buggy_code']], 
        on='ID_Caso', 
        how='left'
    )

    # 4. STEP B: Bring LLM responses for those already processed
    df_final = pd.merge(
        df_base_context, 
        df_llms[['ID_Caso', 'fix_gpt', 'fix_claude', 'fix_gemini']], 
        on='ID_Caso', 
        how='left',
        indicator=True # Adds the _merge column to know where each entry came from
    )
    
    # 5. Create Status column
    df_final['Execution_Status'] = df_final['_merge'].map({
        'both': 'OK - Already resolved in previous round',
        'left_only': 'PENDING - Send to LLMs'
    })
    
    df_final = df_final.drop(columns=['_merge'])
    
    # Reorganize columns in a logical order
    ordered_columns = [
        'ID_Caso', 'Execution_Status', 'Application', 'Resource_Class', 'buggy_code', 
        'fix_gpt', 'fix_claude', 'fix_gemini'
    ]
    df_final = df_final[ordered_columns]

    # 6. Fill empty codes (NaN) in LLM columns with a warning
    for col in ['fix_gpt', 'fix_claude', 'fix_gemini']:
        df_final[col] = df_final[col].fillna("PENDING EXECUTION")

    # 7. Save the final result
    output_path = 'C:/Dissertation/data_bases/05_results/llms_tournament_results_AST_FILTERED.csv'
    df_final.to_csv(output_path, index=False, encoding='utf-8')
    
    # ==========================================
    # MERGE STATISTICS
    # ==========================================
    resolved_count = len(df_final[df_final['Execution_Status'] == 'OK - Already resolved in previous round'])
    pending_count = len(df_final[df_final['Execution_Status'] == 'PENDING - Send to LLMs'])

    print("\n✅ Triple merge completed successfully!")
    print("=" * 60)
    print(f"📊 FINAL SPREADSHEET SUMMARY:")
    print(f" 🗂️ Total instances: {len(df_final)} (Exactly the 290 from AST)")
    print(f" 🟢 {resolved_count} are already complete (with context and old answers).")
    print(f" 🔴 {pending_count} are filled with the original bug and ready to send to the API.")
    print("=" * 60)
    print(f"📁 Spreadsheet ready for LLMs script saved at: {output_path}")

if __name__ == "__main__":
    merge_full_ast_results()