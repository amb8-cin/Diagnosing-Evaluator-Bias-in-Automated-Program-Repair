import pandas as pd
import os

def filter_responses_for_ast():
    print("🎯 Filtering LLM responses for the AST universe...")

    # 1. File paths
    ast_path = 'C:/Dissertacao/data_bases/04_final/holdout_ast.csv' # The ~290 surviving IDs
    responses_300_path = 'C:/Dissertacao/data_bases/04_final/resultados_torneio_llms.csv' # The ready LLM responses
    output_path = 'C:/Dissertacao/data_bases/04_final/results_tournament_llms_FILTERED_AST.csv'

    if not os.path.exists(responses_300_path):
        print(f"❌ ERROR: The file with the 300 bug responses ({responses_300_path}) was not found!")
        print("💡 First, run the script '13_corrigir_bugs.py' to collect LLM responses.")
        return

    # 2. Data reading
    df_ast = pd.read_csv(ast_path)
    df_responses = pd.read_csv(responses_300_path)

    # 3. Performs the join (Inner Join) based on Case_ID
    # This will ensure that the final file has EXACTLY the same IDs as AST,
    # bringing along the 'fix_gpt', 'fix_claude', and 'fix_gemini' columns collected by script 13.
    df_filtered = pd.merge(
        df_ast[['ID_Caso']], 
        df_responses, 
        on='ID_Caso', 
        how='inner'
    )

    # 4. Saves the clean result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_filtered.to_csv(output_path, index=False, encoding='utf-8')

    print("\n✅ Filtering completed successfully!")
    print(f"📊 Total filtered instances: {len(df_filtered)} (Perfect parity with AST).")
    print(f"💾 Generated file ready for AST at: {output_path}")

if __name__ == "__main__":
    filter_responses_for_ast()