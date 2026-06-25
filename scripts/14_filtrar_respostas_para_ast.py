import pandas as pd
import os

def filtrar_respostas_para_ast():
    print("🎯 Filtrando as respostas das LLMs para o universo da AST...")

    # 1. Caminhos dos arquivos
    caminho_ast = 'C:/Dissertacao/data_bases/04_final/holdout_ast.csv' # Os ~290 IDs sobreviventes
    caminho_respostas_300 = 'C:/Dissertacao/data_bases/04_final/resultados_torneio_llms.csv' # As respostas das LLMs prontas
    caminho_saida = 'C:/Dissertacao/data_bases/04_final/resultados_torneio_llms_FILTRADO_AST.csv'

    if not os.path.exists(caminho_respostas_300):
        print(f"❌ ERRO: O arquivo com as respostas dos 300 bugs ({caminho_respostas_300}) não foi encontrado!")
        print("💡 Execute primeiro o script '13_corrigir_bugs.py' para coletar as respostas das LLMs.")
        return

    # 2. Leitura dos dados
    df_ast = pd.read_csv(caminho_ast)
    df_respostas = pd.read_csv(caminho_respostas_300)

    # 3. Faz o cruzamento (Inner Join) baseado no ID_Caso
    # Isso vai garantir que o arquivo final tenha EXATAMENTE os mesmos IDs da AST,
    # trazendo junto as colunas 'fix_gpt', 'fix_claude' e 'fix_gemini' que o script 13 coletou.
    df_filtrado = pd.merge(
        df_ast[['ID_Caso']], 
        df_respostas, 
        on='ID_Caso', 
        how='inner'
    )

    # 4. Salva o resultado limpo
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    df_filtrado.to_csv(caminho_saida, index=False, encoding='utf-8')

    print("\n✅ Filtragem concluída com sucesso!")
    print(f"📊 Total de instâncias filtradas: {len(df_filtrado)} (Paridade perfeita com a AST).")
    print(f"💾 Arquivo gerado pronto para a AST em: {caminho_saida}")

if __name__ == "__main__":
    filtrar_respostas_para_ast()