import pandas as pd

def mesclar_resultados_ast_completo():
    print("🔄 Iniciando o cruzamento triplo de dados...")

    # 1. Caminhos dos arquivos
    caminho_ast = 'C:/Dissertacao/data_bases/ast/holdout_ast.csv' # A LISTA MESTRA (290 IDs)
    caminho_bruto = 'C:/Dissertacao/data_bases/04_final/dataset_sintetico_chatgpt.csv' # O CONTEXTO ORIGINAL
    caminho_llms = 'C:/Dissertacao/data_bases/ast/resultados_torneio_llms.csv' # AS RESPOSTAS ANTIGAS
    
    # 2. Leitura dos dados
    try:
        df_ast = pd.read_csv(caminho_ast)
        df_llms = pd.read_csv(caminho_llms)
    except FileNotFoundError as e:
        print(f"❌ Erro ao ler o arquivo: {e}")
        return

    # Leitura blindada do dataset bruto (caso tenha ponto-e-vírgula)
    try:
        df_bruto = pd.read_csv(caminho_bruto)
    except pd.errors.ParserError:
        df_bruto = pd.read_csv(caminho_bruto, sep=';')

    # 3. PASSO A: Trazer o contexto original (Aplicacao, Recurso, Codigo) para os 290 IDs
    # O arquivo bruto usa 'Codigo_Snippet', vamos renomear para 'codigo_com_bug' para manter o padrão
    df_bruto = df_bruto.rename(columns={'Codigo_Snippet': 'codigo_com_bug'})
    
    # Fazemos um Left Join com o dataset bruto
    df_base_contexto = pd.merge(
        df_ast[['ID_Caso']], 
        df_bruto[['ID_Caso', 'Aplicacao', 'Classe_Recurso', 'codigo_com_bug']], 
        on='ID_Caso', 
        how='left'
    )

    # 4. PASSO B: Trazer as respostas das LLMs para quem já rodou
    df_final = pd.merge(
        df_base_contexto, 
        df_llms[['ID_Caso', 'fix_gpt', 'fix_claude', 'fix_gemini']], 
        on='ID_Caso', 
        how='left',
        indicator=True # Adiciona a coluna _merge para sabermos quem veio de onde
    )
    
    # 5. Criar a coluna de Status
    df_final['Status_Execucao'] = df_final['_merge'].map({
        'both': 'OK - Já resolvido na rodada anterior',
        'left_only': 'FALTA RODAR - Enviar para as LLMs'
    })
    
    df_final = df_final.drop(columns=['_merge'])
    
    # Reorganiza as colunas numa ordem lógica
    colunas_ordenadas = [
        'ID_Caso', 'Status_Execucao', 'Aplicacao', 'Classe_Recurso', 'codigo_com_bug', 
        'fix_gpt', 'fix_claude', 'fix_gemini'
    ]
    df_final = df_final[colunas_ordenadas]

    # 6. Preencher os códigos vazios (NaN) nas colunas das LLMs com o aviso
    for col in ['fix_gpt', 'fix_claude', 'fix_gemini']:
        df_final[col] = df_final[col].fillna("PENDENTE DE EXECUÇÃO")

    # 7. Salvar o resultado final
    caminho_saida = 'C:/Dissertacao/data_bases/ast/resultados_torneio_llms_FILTRADO_AST.csv'
    df_final.to_csv(caminho_saida, index=False, encoding='utf-8')
    
    # ==========================================
    # ESTATÍSTICAS DO CRUZAMENTO
    # ==========================================
    resolvidos = len(df_final[df_final['Status_Execucao'] == 'OK - Já resolvido na rodada anterior'])
    pendentes = len(df_final[df_final['Status_Execucao'] == 'FALTA RODAR - Enviar para as LLMs'])

    print("\n✅ Cruzamento triplo concluído com sucesso!")
    print("=" * 60)
    print(f"📊 RESUMO DA PLANILHA FINAL:")
    print(f" 🗂️ Total de instâncias: {len(df_final)} (Exatamente os 290 da AST)")
    print(f" 🟢 {resolvidos} já estão completos (com contexto e respostas antigas).")
    print(f" 🔴 {pendentes} estão preenchidos com o bug original e prontos para enviar à API.")
    print("=" * 60)
    print(f"📁 Planilha pronta para o script das LLMs salva em: {caminho_saida}")

if __name__ == "__main__":
    mesclar_resultados_ast_completo()