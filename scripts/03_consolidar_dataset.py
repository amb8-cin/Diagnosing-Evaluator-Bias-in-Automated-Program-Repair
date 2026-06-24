import os
import pandas as pd

def consolidar_diffs_em_csv(diretorio_base, arquivo_saida):
    print(f"A iniciar a consolidação do Dataset para IA...\n")
    
    dados_para_csv = []
    
    # Lista apenas as pastas (ignorando ficheiros soltos na raiz)
    pastas = [p for p in os.listdir(diretorio_base) if os.path.isdir(os.path.join(diretorio_base, p))]
    
    casos_processados = 0

    for pasta in pastas:
        caminho_pasta = os.path.join(diretorio_base, pasta)
        diff_file = os.path.join(caminho_pasta, "codigo_diferenca.diff")
        meta_file = os.path.join(caminho_pasta, "metadata.txt")

        # Só processa se o ficheiro diff existir nesta pasta
        if os.path.exists(diff_file):
            with open(diff_file, 'r', encoding='utf-8') as f:
                conteudo_diff = f.read().strip()
            
            # Ignora ficheiros onde não houve diferenças ou que estão vazios
            if not conteudo_diff or "Nenhuma diferença encontrada" in conteudo_diff:
                continue
                
            # Inicializa as variáveis de metadados com valores por defeito
            app_name = pasta
            buggy_file_path = "Desconhecido"
            resource_class = "Desconhecido"
            
            # Extrai o contexto do ficheiro metadata.txt
            if os.path.exists(meta_file):
                with open(meta_file, 'r', encoding='utf-8') as m:
                    linhas_meta = m.readlines()
                    for linha in linhas_meta:
                        if linha.startswith("App:"):
                            app_name = linha.split("App:")[1].strip()
                        elif linha.startswith("Buggy File:"):
                            buggy_file_path = linha.split("Buggy File:")[1].strip()
                        elif linha.startswith("Resource:"):
                            resource_class = linha.split("Resource:")[1].strip()

            # Adiciona a linha (o caso) à nossa lista estruturada
            dados_para_csv.append({
                "ID_Caso": pasta,
                "Aplicacao": app_name,
                "Classe_Recurso": resource_class,
                "Caminho_Ficheiro": buggy_file_path,
                "Codigo_Diff": conteudo_diff,
                "Tem_Fuga_de_Recurso": 1  # 1 indica a presença do bug (útil para IA)
            })
            casos_processados += 1

    # Converter para DataFrame do Pandas e guardar em CSV
    df = pd.DataFrame(dados_para_csv)
    df.to_csv(arquivo_saida, index=False, encoding='utf-8-sig', sep=';') # utf-8-sig ajuda o Excel a ler os acentos corretamente
    
    print(f"✅ Processo concluído com sucesso!")
    print(f"📊 Dataset final gerado com {casos_processados} casos limpos.")
    print(f"📁 Ficheiro guardado em: {arquivo_saida}")

# ==========================================
# COMO EXECUTAR
# ==========================================
if __name__ == "__main__":
    # A pasta onde estão os seus casos extraídos
    PASTA_DATASET = 'C:/Dissertacao/data_bases/02_extracted'
    
    # O ficheiro final que será gerado (vai ficar fora da pasta dos repositórios para não misturar)
    ARQUIVO_CSV_FINAL = 'C:/Dissertacao/data_bases/01_raw/dataset_extraido_droidleaks.csv'
    
    consolidar_diffs_em_csv(PASTA_DATASET, ARQUIVO_CSV_FINAL)