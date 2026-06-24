import pandas as pd

def extrair_versoes_do_diff(codigo_diff):
    """
    Desmonta o diff e reconstrói o código original com bug (1) 
    e o código corrigido (0), limpando os sinais de diff.
    """
    if not isinstance(codigo_diff, str):
        return "", ""
        
    linhas_buggy = []
    linhas_fix = []
    
    # Ignorar as primeiras linhas de cabeçalho do diff (--- e +++)
    linhas = codigo_diff.split('\n')
    for linha in linhas:
        if linha.startswith('---') or linha.startswith('+++') or linha.startswith('@@'):
            continue
            
        # Se a linha começa com '-', pertence SÓ à versão com bug
        if linha.startswith('-'):
            linhas_buggy.append(linha[1:]) # Remove o '-'
            
        # Se a linha começa com '+', pertence SÓ à versão corrigida
        elif linha.startswith('+'):
            linhas_fix.append(linha[1:]) # Remove o '+'
            
        # Se começar com espaço, pertence a AMBAS as versões (é o contexto)
        elif linha.startswith(' '):
            linhas_buggy.append(linha[1:])
            linhas_fix.append(linha[1:])
            
        # Se não tiver prefixo (as vezes acontece por formatação), assume contexto
        else:
            linhas_buggy.append(linha)
            linhas_fix.append(linha)
            
    return '\n'.join(linhas_buggy).strip(), '\n'.join(linhas_fix).strip()

def gerar_dataset_balanceado(caminho_csv_entrada, caminho_csv_saida):
    print("A iniciar a extração dos '0s' e balanceamento do Dataset...\n")
    
    df = pd.read_csv(caminho_csv_entrada, sep=None, engine='python', encoding='utf-8-sig')
    novos_dados = []
    
    casos_processados = 0
    
    for index, row in df.iterrows():
        diff = row['Codigo_Diff']
        codigo_buggy, codigo_fix = extrair_versoes_do_diff(diff)
        
        # Só adiciona se a extração funcionou
        if codigo_buggy and codigo_fix:
            # 1. Adiciona a versão COM BUG (Label = 1)
            novos_dados.append({
                "ID_Caso": f"{row['ID_Caso']}_BUG",
                "Aplicacao": row['Aplicacao'],
                "Classe_Recurso": row['Classe_Recurso'],
                "Codigo_Snippet": codigo_buggy,
                "Tem_Fuga_de_Recurso": 1  # 🔴 DOENTE
            })
            
            # 2. Adiciona a versão CORRIGIDA (Label = 0)
            novos_dados.append({
                "ID_Caso": f"{row['ID_Caso']}_FIX",
                "Aplicacao": row['Aplicacao'],
                "Classe_Recurso": row['Classe_Recurso'],
                "Codigo_Snippet": codigo_fix,
                "Tem_Fuga_de_Recurso": 0  # 🟢 SAUDÁVEL
            })
            
            casos_processados += 1
            
    # Cria o novo DataFrame
    df_balanceado = pd.DataFrame(novos_dados)
    
    # Guarda o novo ficheiro (usando sep=';' para abrir bem no seu Excel)
    df_balanceado.to_csv(caminho_csv_saida, index=False, sep=';', encoding='utf-8-sig')
    
    print("✅ Processo concluído com sucesso!")
    print(f"📊 Processámos {casos_processados} Diffs originais.")
    print(f"⚖️ O novo dataset tem agora {len(df_balanceado)} casos (50% com bug, 50% saudáveis)!")
    print(f"📁 Ficheiro guardado em: {caminho_csv_saida}")

# ==========================================
# COMO EXECUTAR
# ==========================================
if __name__ == "__main__":
    # Vamos usar o CSV principal que criámos há pouco
    CSV_ORIGINAL = 'C:/Dissertacao/data_bases/dataset_treino_droidleaks.csv'
    
    # O novo ficheiro com o dobro do tamanho e os "0s" incluídos
    CSV_BALANCEADO = 'C:/Dissertacao/data_bases/data_bases/03_processed/dataset_balanceado.csv'
    
    gerar_dataset_balanceado(CSV_ORIGINAL, CSV_BALANCEADO)