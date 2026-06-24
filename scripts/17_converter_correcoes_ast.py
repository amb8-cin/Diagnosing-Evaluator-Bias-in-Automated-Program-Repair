import javalang
import pandas as pd
import re

def tentar_fazer_parse(codigo_limpo):
    """
    Tenta empacotar o código em diferentes níveis (Classe, Método ou Statements)
    até o javalang conseguir montar a árvore (AST).
    """
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

def extrair_caminhos_ast(codigo_java):
    """
    Extrai as assinaturas sintáticas do código, limpando sujeiras de LLMs.
    """
    codigo_str = str(codigo_java)
    
    codigo_limpo = codigo_str.replace("【", "").replace("】", "")
    codigo_limpo = re.sub(r"^```java|```$", "", codigo_limpo, flags=re.MULTILINE).strip()
    
    arvore = tentar_fazer_parse(codigo_limpo)
    
    if arvore is None:
        return "ERRO_SINTAXE"

    features_extraidas = []
    for caminho, no in arvore:
        if isinstance(no, javalang.tree.TryStatement):
            features_extraidas.append("TryStatement")
            if no.resources and len(no.resources) > 0:
                features_extraidas.append("HasResources_ImplicitClose")
            if getattr(no, 'finally_block', None):
                features_extraidas.append("HasFinallyBlock")
                
        elif isinstance(no, javalang.tree.MethodInvocation):
            if no.member == "close":
                features_extraidas.append("MethodCall_close") 
            elif no.member == "disconnect":
                features_extraidas.append("MethodCall_disconnect") 
            elif no.member == "release":
                features_extraidas.append("MethodCall_release") 
            elif no.member == "recycle":
                features_extraidas.append("MethodCall_recycle") 
            elif no.member == "free":
                features_extraidas.append("MethodCall_free") 
            elif no.member == "destroy":
                features_extraidas.append("MethodCall_destroy") 
            else:
                features_extraidas.append("MethodCall_Other")
                
        elif isinstance(no, javalang.tree.CatchClause):
            features_extraidas.append("CatchClause")
            
    if not features_extraidas:
        return "SEM_FEATURES_RELEVANTES"
        
    return " ".join(features_extraidas)

def processar_respostas_torneio(caminho_csv_entrada, caminho_csv_saida):
    print(f"📄 Lendo as respostas do torneio em: {caminho_csv_entrada}")
    
    try:
        df = pd.read_csv(caminho_csv_entrada)
    except Exception as e:
        print(f"❌ Erro ao ler o CSV: {e}")
        return

    print(f"📊 Total de bugs analisados pelas IAs: {len(df)}")
    print("🌲 Iniciando a conversão dos códigos para Árvores Sintáticas (AST)...\n")

    # Mapeamento: Coluna Original de Código -> Nova Coluna de Features AST
    colunas_llms = {
        'fix_claude': 'AST_Claude',
        'fix_gpt': 'AST_GPT',
        'fix_gemini': 'AST_Gemini'
    }

    for col_codigo, col_ast in colunas_llms.items():
        if col_codigo in df.columns:
            print(f"   -> Extraindo estruturas do modelo: {col_codigo}...")
            # Extrai as features mas NÃO deleta as linhas com erro. 
            # Erro de sintaxe significa que a IA falhou no reparo!
            df[col_ast] = df[col_codigo].apply(extrair_caminhos_ast)
        else:
            print(f"   ⚠️ Coluna {col_codigo} não encontrada no CSV!")

    # Selecionamos apenas as colunas que importam para o oráculo
    colunas_finais = ['ID_Caso'] + list(colunas_llms.values())
    df_ast = df[colunas_finais].copy()

    df_ast.to_csv(caminho_csv_saida, index=False)
    
    print("\n✅ Sucesso! Todas as respostas foram convertidas.")
    print(f"💾 Arquivo de features gerado em: {caminho_csv_saida}")
    print("🚀 Agora você já pode rodar o script 'torneio_llms_ast.py'!")

# =====================================================================
if __name__ == "__main__":
    # Arquivo original com os códigos em Java puro
    ARQUIVO_ENTRADA = "C:/Dissertacao/data_bases/ast/resultados_torneio_llms_FILTRADO_AST.csv" 
    
    # Arquivo novo que será consumido pelo validador oráculo
    ARQUIVO_SAIDA = "C:/Dissertacao/data_bases/ast/respostas_llms_ast.csv"
    
    processar_respostas_torneio(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)