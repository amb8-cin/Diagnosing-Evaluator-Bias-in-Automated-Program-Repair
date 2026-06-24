import pandas as pd
import os

def alinhar_split_ast(caminho_ast_completo, caminho_treino_real, caminho_holdout_llm, pasta_saida):
    print("🔄 Alinhando o Split da AST com os dados oficiais do Torneio...")
    
    # 1. Carrega os arquivos
    df_ast = pd.read_csv(caminho_ast_completo)
    df_treino_oficial = pd.read_csv(caminho_treino_real, sep=';')
    df_holdout_oficial = pd.read_csv(caminho_holdout_llm) # os 300 bugs das LLMs
    
    # 2. Cria o ID do Par na AST para conseguir mapear
    df_ast['ID_Par'] = df_ast['ID_Caso'].apply(lambda x: str(x).rsplit('_', 1)[0])
    
    # Cria o ID do par nos arquivos oficiais para servir de chave
    df_treino_oficial['ID_Par'] = df_treino_oficial['ID_Caso'].apply(lambda x: str(x).rsplit('_', 1)[0])
    df_holdout_oficial['ID_Par'] = df_holdout_oficial['ID_Caso'].apply(lambda x: str(x).replace('_BUG', '').replace('_FIX', ''))
    
    lista_pares_treino = df_treino_oficial['ID_Par'].unique()
    lista_pares_holdout = df_holdout_oficial['ID_Par'].unique()
    
    # 3. Separa a AST usando as listas oficiais (Garante paridade e impede Data Leakage)
    df_ast_treino = df_ast[df_ast['ID_Par'].isin(lista_pares_treino)].drop(columns=['ID_Par'])
    df_ast_teste = df_ast[df_ast['ID_Par'].isin(lista_pares_holdout)].drop(columns=['ID_Par'])
    
    # 4. Cria o holdout da AST (Apenas os bugs do grupo de teste)
    df_ast_holdout_bugs = df_ast_teste[df_ast_teste['Tem_Fuga_de_Recurso'] == 1]
    
    # Salva os arquivos perfeitamente alinhados
    os.makedirs(pasta_saida, exist_ok=True)
    df_ast_treino.to_csv(os.path.join(pasta_saida, "dataset_ast_treino.csv"), index=False)
    df_ast_teste.to_csv(os.path.join(pasta_saida, "dataset_ast_teste.csv"), index=False)
    df_ast_holdout_bugs.to_csv(os.path.join(pasta_saida, "holdout_ast.csv"), index=False)
    
    print("\n✅ Alinhamento concluído! Seus modelos agora estão pareados cientificamente.")
    print(f"📈 Treino AST: {len(df_ast_treino)} linhas | 🎯 Teste/Holdout AST: {len(df_ast_holdout_bugs)} bugs.")

if __name__ == "__main__":
    AST_COMPLETA = "C:/Dissertacao/data_bases/04_final/dataset_ast_completo.csv"
    TREINO_REAL = "C:/Dissertacao/data_bases/04_final/treino_validador_final.csv"
    HOLDOUT_LLM = "C:/Dissertacao/data_bases/04_final/holdout_300_bugs_llm.csv"
    PASTA_DESTINO = "C:/Dissertacao/data_bases/04_final/"
    
    alinhar_split_ast(AST_COMPLETA, TREINO_REAL, HOLDOUT_LLM, PASTA_DESTINO)