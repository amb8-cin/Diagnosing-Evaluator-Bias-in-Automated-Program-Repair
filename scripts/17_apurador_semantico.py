import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🤖 Iniciando o Apurador Semântico (GraphCodeBERT)...")

    # ==========================================
    # 1. TREINAR O "JUIZ" (RANDOM FOREST)
    # ==========================================
    print("📂 Carregando a base de conhecimento do Juiz...")
    caminho_embeddings_base = 'C:/Dissertacao/data_bases/graphcodebert/embeddings_graphcodebert.csv'
    df_base = pd.read_csv(caminho_embeddings_base)
    
    # --- ATUALIZAÇÃO: Removendo o ID_Par para não causar erro de string ---
    colunas_para_remover = ['Tem_Fuga_de_Recurso']
    if 'ID_Par' in df_base.columns:
        colunas_para_remover.append('ID_Par')
        
    X_train = df_base.drop(columns=colunas_para_remover).values
    y_train = df_base['Tem_Fuga_de_Recurso'].values

    print(f"🌲 Treinando o Juiz (Random Forest) com {len(X_train)} exemplos...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # ==========================================
    # 2. PREPARAR O EXTRATOR NEURAL
    # ==========================================
    print("\n📥 Carregando o modelo extrator da Microsoft...")
    nome_modelo = "microsoft/graphcodebert-base"
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    model = AutoModel.from_pretrained(nome_modelo)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def extrair_embedding(codigo):
        # Se a LLM não respondeu ou deu erro, retornamos um vetor de zeros
        if pd.isna(codigo) or str(codigo).strip() == "" or str(codigo).startswith("ERRO"):
            return np.zeros(768)
            
        inputs = tokenizer(str(codigo), return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0, 0, :].cpu().numpy()

    # ==========================================
    # 3. AVALIAR AS LLMs
    # ==========================================
    caminho_torneio = 'C:/Dissertacao/data_bases/05_results/resultados_torneio_llms.csv'
    print(f"\n📂 Lendo os resultados do torneio: {caminho_torneio}")
    df_llms = pd.read_csv(caminho_torneio)
    
    total_casos = len(df_llms)
    resultados = {"GPT": 0, "Claude": 0, "Gemini": 0}

    colunas_modelos = {'fix_gpt': 'GPT', 'fix_claude': 'Claude', 'fix_gemini': 'Gemini'}

    for col_df, nome_llm in colunas_modelos.items():
        print(f"\n🧠 Extraindo semântica e avaliando as respostas do {nome_llm}...")
        
        # Tratamento de erro caso alguma coluna da LLM não exista no CSV
        if col_df not in df_llms.columns:
            print(f"⚠️ Aviso: Coluna '{col_df}' não encontrada no CSV. Pulando {nome_llm}.")
            continue
            
        codigos_gerados = df_llms[col_df].tolist()
        
        vetores_llm = []
        for i, codigo in enumerate(codigos_gerados):
            if i % 50 == 0 and i > 0:
                print(f"   -> Analisados {i}/{total_casos}...")
            vetores_llm.append(extrair_embedding(codigo))
            
        # O Juiz avalia todos os códigos dessa LLM de uma vez
        predicoes = rf.predict(vetores_llm)
        
        # Conta quantos o modelo previu como 0 (Sem Bug / Corrigido)
        corrigidos = np.sum(predicoes == 0)
        resultados[nome_llm] = corrigidos

    # ==========================================
    # 4. EXIBIR RESULTADOS FINAIS
    # ==========================================
    print("\n" + "="*60)
    print("🏆 RESULTADOS DO APURADOR SEMÂNTICO (GraphCodeBERT)")
    print("="*60)
    
    # Ordena do maior para o menor
    ranking = sorted(resultados.items(), key=lambda item: item[1], reverse=True)
    
    for nome_llm, corrigidos in ranking:
        taxa = (corrigidos / total_casos) * 100
        print(f"-> {nome_llm}: Corrigiu {corrigidos}/{total_casos} bugs ({taxa:.2f}%)")
    print("="*60)

    # ==========================================
    # 4. GERAR O GRÁFICO DO PLACAR FINAL
    # ==========================================
    if resultados:
        print("\n🎨 Gerando gráfico do placar final do torneio...")
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # 1. Mapear os nomes curtos do dicionário para os nomes formais da dissertação
        nomes_oficiais = {
            "Claude": "Claude 4.6 Opus",
            "GPT": "GPT-5.4",
            "Gemini": "Gemini 3.1 Pro"
        }
        
        # 2. Transformar os números absolutos em porcentagem (%)
        resultados_percentuais = {}
        for nome_curto, acertos in resultados.items():
            taxa_pct = (acertos / total_casos) * 100
            nome_formal = nomes_oficiais.get(nome_curto, nome_curto)
            resultados_percentuais[nome_formal] = taxa_pct
        
        # 3. Ordenar os resultados do maior para o menor (com base na porcentagem)
        resultados_ordenados = dict(sorted(resultados_percentuais.items(), key=lambda item: item[1], reverse=True))
        
        nomes = list(resultados_ordenados.keys())
        taxas = list(resultados_ordenados.values())
        
        # Paleta de cores exata (Verde, Laranja, Azul)
        cores = ['#2CA02C', '#FF7F0E', '#1F77B4']
        
        # Criar barras sem bordas para ficar igual ao exemplo
        barras = ax.bar(nomes, taxas, color=cores, width=0.5, edgecolor='none')
        
        # Atualizar rótulos e títulos (Removido o LFV, pois agora é Semântico)
        ax.set_ylabel('Taxa de Sucesso Semântico %', fontsize=12, fontweight='bold')
        ax.set_title('Placar Final: Eficácia das LLMs no Reparo (Validação GraphCodeBERT)', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        
        # Adicionar os rótulos de porcentagem em cima das barras
        for barra in barras:
            altura = barra.get_height()
            ax.annotate(f'{altura:.2f}%',
                        xy=(barra.get_x() + barra.get_width() / 2, altura),
                        xytext=(0, 3),  # 3 points de deslocamento vertical
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        nome_arquivo_grafico = 'Placar_Final_GraphCodeBERT.png'
        plt.savefig(nome_arquivo_grafico, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico salvo com sucesso como '{nome_arquivo_grafico}'")

if __name__ == "__main__":
    main()