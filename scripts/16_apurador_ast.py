import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🏆 Iniciando o Torneio das LLMs (Validador AST)...\n")

    # 1. CARREGAR O ORÁCULO E O VETORIZADOR
    print("1. Carregando o modelo AST e o vetorizador...")
    try:
        modelo_ast = joblib.load('C:/Dissertacao/trained_models/validador_ast_rf.pkl')
        vectorizer_ast = joblib.load('C:/Dissertacao/trained_models/vectorizer_ast.pkl')
    except FileNotFoundError:
        print("❌ ERRO: Arquivos do modelo não encontrados. Verifique o caminho.")
        return

    # 2. CARREGAR OS DADOS DAS LLMs
    # Substitua pelo caminho real do seu CSV que contém as respostas/features das LLMs
    caminho_respostas = 'C:/Dissertacao/data_bases/ast/respostas_llms_ast.csv' 
    print(f"2. Lendo as respostas das LLMs em: {caminho_respostas}...")
    
    try:
        df_llms = pd.read_csv(caminho_respostas)
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo {caminho_respostas} não encontrado.")
        return

    total_bugs = len(df_llms)
    print(f"   -> Total de instâncias analisadas: {total_bugs}\n")

    # 3. VETORIZAR E AVALIAR CADA MODELO
    # IMPORTANTE: Coloque aqui o nome exato das colunas onde estão as features AST de cada IA
    modelos_ia = {
        'Claude 4.6 Opus': 'AST_Claude',
        'GPT-5.4': 'AST_GPT',
        'Gemini 3.1 Pro': 'AST_Gemini'
    }

    resultados = {}

    print("3. Processando previsões do Oráculo...")
    for nome_ia, coluna_ast in modelos_ia.items():
        if coluna_ast not in df_llms.columns:
            print(f"   ⚠️ Coluna {coluna_ast} não encontrada no CSV. Pulando {nome_ia}...")
            continue
            
        # Tratar nulos caso alguma IA não tenha gerado código válido
        features_ia = df_llms[coluna_ast].fillna("SEM_FEATURES")
        
        # Vetorizar usando o vocabulário de 9 nós aprendido no treino
        X_ia_vetorizado = vectorizer_ast.transform(features_ia)
        
        # Prever (0 = Corrigido, 1 = Bug)
        previsoes = modelo_ast.predict(X_ia_vetorizado)
        
        # Contar quantos zeros (códigos corrigidos) a IA conseguiu
        sucessos = (previsoes == 0).sum()
        taxa_sucesso = (sucessos / total_bugs) * 100
        
        resultados[nome_ia] = taxa_sucesso
        print(f"   -> {nome_ia}: Corrigiu {sucessos}/{total_bugs} bugs ({taxa_sucesso:.2f}%)")

    # 4. GERAR O GRÁFICO DO PLACAR FINAL
    if resultados:
        print("\n🎨 Gerando gráfico do placar final do torneio...")
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Ordenar os resultados do maior para o menor
        resultados_ordenados = dict(sorted(resultados.items(), key=lambda item: item[1], reverse=True))
        
        nomes = list(resultados_ordenados.keys())
        taxas = list(resultados_ordenados.values())
        
        # Paleta de cores moderna
        cores = ['#2ca02c', '#ff7f0e', '#1f77b4']
        
        barras = ax.bar(nomes, taxas, color=cores, width=0.5)
        
        ax.set_ylabel('Taxa de Sucesso (LFV) %', fontsize=12, fontweight='bold')
        ax.set_title('Placar Final: Eficácia das LLMs no Reparo (Validação AST)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        
        # Adicionar os rótulos de porcentagem em cima das barras
        for barra in barras:
            altura = barra.get_height()
            ax.annotate(f'{altura:.2f}%',
                        xy=(barra.get_x() + barra.get_width() / 2, altura),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('Figura_Placar_Torneio_AST.png', dpi=300)
        print("✅ Gráfico salvo como 'Figura_Placar_Torneio_AST.png'")
        plt.show()

if __name__ == "__main__":
    main()