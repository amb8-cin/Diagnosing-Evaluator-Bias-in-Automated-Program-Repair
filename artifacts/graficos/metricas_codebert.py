import matplotlib.pyplot as plt
import numpy as np

def gerar_grafico_metricas_graphcodebert():
    print("📊 Gerando o gráfico de métricas (GraphCodeBERT)...")
    
    # Nomes das métricas
    metricas = ['Precision\n(Precisão)', 'Recall\n(Sensibilidade)', 'F1-Score\n(Equilíbrio)']
    
    # VALORES REAIS DO SEU RELATÓRIO DE CLASSIFICAÇÃO
    # Classe 0: 0.84, 0.80, 0.82
    # Classe 1: 0.81, 0.85, 0.83
    valores_classe_0 = [0.84, 0.80, 0.82] 
    valores_classe_1 = [0.81, 0.85, 0.83] 
    
    x = np.arange(len(metricas))
    width = 0.35 
    
    # Estilo branco com grade suave
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Cores acadêmicas (Azul e Laranja Sóbrio)
    rects1 = ax.bar(x - width/2, valores_classe_0, width, label='Código Seguro (0)', color='#4C72B0')
    rects2 = ax.bar(x + width/2, valores_classe_1, width, label='Com Bug (1)', color='#DD8452')
    
    # Configurações de rótulos
    ax.set_ylabel('Pontuação (0.0 a 1.0)', fontsize=12, weight='bold')
    ax.set_title('Métricas de Avaliação do Validador Semântico GraphCodeBERT', fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas, fontsize=11, weight='bold')
    
    ax.set_ylim(0, 1.0) # Escala correta de 0 a 1
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    
    # Adicionar os valores em cima das barras
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=11, weight='bold')
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=11, weight='bold')
    
    plt.tight_layout()
    
    # Salvar em 300 DPI
    nome_arquivo = 'Metricas_Avaliacao_GraphCodeBERT.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    
    print(f"✅ Sucesso! Gráfico salvo como: {nome_arquivo}")

if __name__ == "__main__":
    gerar_grafico_metricas_graphcodebert()