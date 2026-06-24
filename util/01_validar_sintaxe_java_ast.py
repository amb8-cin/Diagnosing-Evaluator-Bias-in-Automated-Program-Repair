import javalang
import pandas as pd
import re

# =====================================================================
# FUNÇÕES DE VALIDAÇÃO (As mesmas que usamos antes)
# =====================================================================
def tentar_fazer_parse(codigo_limpo):
    try:
        return javalang.parse.parse(codigo_limpo)
    except Exception:
        pass

    linhas = codigo_limpo.split('\n')
    imports = []
    corpo = []
    for linha in linhas:
        if linha.strip().startswith("import ") or linha.strip().startswith("package "):
            imports.append(linha)
        else:
            corpo.append(linha)
            
    str_imports = "\n".join(imports)
    str_corpo = "\n".join(corpo)

    codigo_classe = f"{str_imports}\npublic class DummyClass {{\n{str_corpo}\n}}"
    try:
        return javalang.parse.parse(codigo_classe)
    except Exception:
        pass

    codigo_metodo = f"{str_imports}\npublic class DummyClass {{\n public void dummyMethod() throws Exception {{\n{str_corpo}\n}}\n}}"
    try:
        return javalang.parse.parse(codigo_metodo)
    except Exception:
        pass

    codigo_chaves = codigo_metodo + "\n}\n}\n}"
    try:
        return javalang.parse.parse(codigo_chaves)
    except Exception:
        return None

def tem_erro_sintaxe(codigo_java):
    codigo_str = str(codigo_java)
    codigo_limpo = codigo_str.replace("【", "").replace("】", "")
    codigo_limpo = re.sub(r"^```java|```$", "", codigo_limpo, flags=re.MULTILINE).strip()
    
    arvore = tentar_fazer_parse(codigo_limpo)
    return arvore is None

# =====================================================================
# LÓGICA DE INSPEÇÃO
# =====================================================================
def inspecionar_falhas(caminho_csv_entrada, caminho_txt_saida, coluna_codigo):
    print(f"🔎 Lendo CSV para caçar os erros: {caminho_csv_entrada}")
    
    try:
        df = pd.read_csv(caminho_csv_entrada)
    except pd.errors.ParserError:
        df = pd.read_csv(caminho_csv_entrada, sep=';')

    print("🕵️‍♂️ Identificando os snippets problemáticos...\n")
    
    snippets_com_erro = []
    
    # Itera pelas linhas do dataframe
    for index, row in df.iterrows():
        codigo = row[coluna_codigo]
        if tem_erro_sintaxe(codigo):
            snippets_com_erro.append((index, codigo))

    total_erros = len(snippets_com_erro)
    print(f"⚠️ Foram encontrados {total_erros} snippets com erro de sintaxe.\n")
    
    if total_erros == 0:
        print("Tudo perfeito! Nenhum erro encontrado.")
        return

    # Imprime os 3 primeiros no terminal para matar a curiosidade rápida
    print("-" * 50)
    print("👀 PREVIEW DOS 3 PRIMEIROS ERROS:")
    print("-" * 50)
    for i in range(min(3, total_erros)):
        idx, cod = snippets_com_erro[i]
        print(f"[LINHA {idx + 2} DO CSV]") # +2 porque o pandas conta do 0 e o CSV tem cabeçalho
        print(cod)
        print("\n" + "="*50 + "\n")

    # Salva todos em um arquivo TXT para análise
    with open(caminho_txt_saida, "w", encoding="utf-8") as f:
        f.write(f"RELATÓRIO DE INSPEÇÃO: {total_erros} SNIPPETS REJEITADOS\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, cod in snippets_com_erro:
            f.write(f"--- ERRO NA LINHA {idx + 2} DO CSV original ---\n")
            f.write(str(cod) + "\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
    print(f"📁 Um relatório completo com todos os {total_erros} códigos quebrados foi salvo em:")
    print(f"👉 {caminho_txt_saida}")

# =====================================================================
# CONFIGURAÇÃO
# =====================================================================
if __name__ == "__main__":
    ARQUIVO_ENTRADA = "C:/Dissertacao/data_bases/04_final/dataset_sintetico_chatgpt.csv" 
    ARQUIVO_TXT_SAIDA = "C:/Dissertacao/artifacts/erros_inspecao.txt"
    COLUNA_CODIGO = 'Codigo_Snippet' 
    
    inspecionar_falhas(ARQUIVO_ENTRADA, ARQUIVO_TXT_SAIDA, COLUNA_CODIGO)