import javalang
import pandas as pd
import re

def tentar_fazer_parse(codigo_limpo):
    """
    Tenta empacotar o código em diferentes níveis (Classe, Método ou Statements)
    até o javalang conseguir montar a árvore (AST).
    """
    # 1. Tenta fazer o parse como está (caso já seja uma classe completa gerada pelo LLM)
    try:
        return javalang.parse.parse(codigo_limpo)
    except Exception:
        pass

    # 2. Separa os imports do resto do código.
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

    # 3. TENTATIVA 2: O snippet é um método solto? 
    codigo_classe = f"{str_imports}\npublic class DummyClass {{\n{str_corpo}\n}}"
    try:
        return javalang.parse.parse(codigo_classe)
    except Exception:
        pass

    # 4. TENTATIVA 3: O snippet é apenas uma lógica solta (Statements)?
    codigo_metodo = f"{str_imports}\npublic class DummyClass {{\n public void dummyMethod() throws Exception {{\n{str_corpo}\n}}\n}}"
    try:
        return javalang.parse.parse(codigo_metodo)
    except Exception:
        pass

    # 5. TENTATIVA 4 (Salva-vidas): O LLM cortou o código antes de fechar as chaves?
    codigo_chaves = codigo_metodo + "\n}\n}\n}"
    try:
        return javalang.parse.parse(codigo_chaves)
    except Exception:
        return None # Desiste, a sintaxe está realmente destruída

def extrair_caminhos_ast(codigo_java):
    """
    Extrai as assinaturas sintáticas do código, limpando sujeiras de LLMs.
    """
    codigo_str = str(codigo_java)
    
    # 1. Limpeza agressiva de artefatos de LLMs (Markdown, colchetes, etc)
    codigo_limpo = codigo_str.replace("【", "").replace("】", "")
    codigo_limpo = re.sub(r"^```java|```$", "", codigo_limpo, flags=re.MULTILINE).strip()
    
    # 2. Busca a árvore sintática usando as tentativas progressivas
    arvore = tentar_fazer_parse(codigo_limpo)
    
    if arvore is None:
        return "ERRO_SINTAXE"

    # 3. Extração das features
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

def gerar_dataset_ast_do_csv(caminho_csv_entrada, caminho_csv_saida, coluna_id, coluna_codigo, coluna_label):
    print(f"📄 Lendo os snippets do arquivo: {caminho_csv_entrada}")
    
    try:
        df = pd.read_csv(caminho_csv_entrada)
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(caminho_csv_entrada, sep=';')
        except Exception:
            print("⚠️ Formatação irregular detectada. Usando motor Python de leitura...")
            df = pd.read_csv(caminho_csv_entrada, sep=None, engine='python', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(caminho_csv_entrada, encoding='latin1', sep=';')

    print(f"📊 Total de registros lidos do CSV: {len(df)}")
    
    if coluna_id not in df.columns or coluna_codigo not in df.columns or coluna_label not in df.columns:
        print(f"❌ ERRO CRÍTICO: As colunas '{coluna_id}', '{coluna_codigo}' ou '{coluna_label}' não foram encontradas!")
        print(f"Colunas disponíveis no arquivo: {list(df.columns)}")
        return

    print("🌲 Extraindo as Árvores de Sintaxe Abstrata (AST)... Isso pode levar alguns segundos.")
    df['AST_Features'] = df[coluna_codigo].astype(str).apply(extrair_caminhos_ast)
    
    erros = len(df[df['AST_Features'] == "ERRO_SINTAXE"])
    print(f"⚠️ Snippets ignorados por erro de sintaxe intransponível: {erros}")
    
    # 1. Remove as linhas individuais com erro
    df = df[df['AST_Features'] != "ERRO_SINTAXE"]
    
    # =====================================================================
    # 🛡️ NOVA BLINDAGEM: REMOÇÃO DE PARES ÓRFÃOS (Garante 50% / 50%)
    # =====================================================================
    print("⚖️ Verificando paridade dos dados (removendo casos órfãos)...")
    
    # Cria a coluna temporária do ID do par tirando o sufixo _BUG ou _FIX
    df['ID_Par_Temp'] = df[coluna_id].apply(lambda x: str(x).rsplit('_', 1)[0])
    
    # Conta os sobreviventes. Pares perfeitos devem ter exatamente 2 ocorrências
    contagem_pares = df['ID_Par_Temp'].value_counts()
    pares_completos = contagem_pares[contagem_pares == 2].index
    
    # Filtra mantendo apenas as linhas cujos pares estão 100% completos
    df = df[df['ID_Par_Temp'].isin(pares_completos)]
    # =====================================================================

    # Filtramos para salvar mantendo a coluna de ID intacta
    df_final = df[[coluna_id, 'AST_Features', coluna_label]].copy()
    
    # Salva o novo CSV completo
    df_final.to_csv(caminho_csv_saida, index=False)
    
    # Métricas para a sua paz de espírito no console
    total_bugs = len(df_final[df_final[coluna_label] == 1])
    total_fix = len(df_final[df_final[coluna_label] == 0])
    
    print(f"\n✅ Extração e Alinhamento concluídos com sucesso!")
    print(f"📊 Dataset estrutural balanceado: {len(df_final)} linhas válidas.")
    print(f"   ↳ 🔴 Classe 1 (Com Bug): {total_bugs} exemplos")
    print(f"   ↳ 🟢 Classe 0 (Corrigido): {total_fix} exemplos")
    print(f"💾 Arquivo salvo em: {caminho_csv_saida}")

if __name__ == "__main__":
    ARQUIVO_ENTRADA = "C:/Dissertacao/data_bases/04_final/dataset_sintetico_chatgpt.csv" 
    ARQUIVO_SAIDA = "C:/Dissertacao/data_bases/04_final/dataset_ast_completo.csv"
    
    COLUNA_ID = 'ID_Caso'
    COLUNA_CODIGO = 'Codigo_Snippet' 
    COLUNA_LABEL = 'Tem_Fuga_de_Recurso'
    
    gerar_dataset_ast_do_csv(ARQUIVO_ENTRADA, ARQUIVO_SAIDA, COLUNA_ID, COLUNA_CODIGO, COLUNA_LABEL)