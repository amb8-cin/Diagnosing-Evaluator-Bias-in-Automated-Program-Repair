import os
import difflib

def gerar_diffs_dataset(diretorio_base):
    print("A iniciar a geração de Diffs para o DroidLeaks...\n")
    
    # Lista todas as pastas dentro do dataset
    pastas = [p for p in os.listdir(diretorio_base) if os.path.isdir(os.path.join(diretorio_base, p))]
    
    casos_processados = 0
    erros = 0

    for pasta in pastas:
        caminho_pasta = os.path.join(diretorio_base, pasta)
        buggy_file = os.path.join(caminho_pasta, "BuggyCode.java")
        fix_file = os.path.join(caminho_pasta, "GroundTruth_Fix.java")
        output_diff = os.path.join(caminho_pasta, "codigo_diferenca.diff")

        # Só processa se a pasta tiver os dois ficheiros Java
        if os.path.exists(buggy_file) and os.path.exists(fix_file):
            try:
                # Lemos os ficheiros (usando errors='ignore' para evitar problemas com caracteres antigos)
                with open(buggy_file, 'r', encoding='utf-8', errors='ignore') as f_buggy:
                    buggy_lines = f_buggy.readlines()
                    
                with open(fix_file, 'r', encoding='utf-8', errors='ignore') as f_fix:
                    fix_lines = f_fix.readlines()

                # Gerar o Diff (com 3 linhas de contexto para cima e para baixo do erro)
                diff = difflib.unified_diff(
                    buggy_lines, 
                    fix_lines, 
                    fromfile='BuggyCode.java (Com Erro)', 
                    tofile='GroundTruth_Fix.java (Corrigido)',
                    n=3  # Altere este número se quiser mais ou menos linhas de contexto
                )
                
                linhas_diff = list(diff)
                
                # Guarda o resultado no novo ficheiro .diff
                with open(output_diff, 'w', encoding='utf-8') as f_out:
                    if len(linhas_diff) == 0:
                        f_out.write("Nenhuma diferença encontrada. Os ficheiros são idênticos.")
                    else:
                        f_out.writelines(linhas_diff)
                
                casos_processados += 1
                
            except Exception as e:
                print(f"⚠️ Erro ao processar o caso {pasta}: {e}")
                erros += 1

    print(f"\n✅ Processo concluído!")
    print(f"📊 Diffs gerados com sucesso: {casos_processados} casos.")
    if erros > 0:
        print(f"❌ Erros encontrados: {erros}")

# ==========================================
# COMO EXECUTAR
# ==========================================
if __name__ == "__main__":
    # Caminho exato da sua pasta com os casos extraídos
    PASTA_DATASET = 'C:\Dissertacao\data_bases\02_extracted'
    
    gerar_diffs_dataset(PASTA_DATASET)