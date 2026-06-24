import matplotlib.pyplot as plt
import numpy as np

def gerar_grafico_metricas():
    print("📊 A gerar o gráfico de barras das métricas (Figura Y)...")
    
    # Nomes das métricas
    metricas = ['Precision\n(Precisão)', 'Recall\n(Sensibilidade)', 'F1-Score\n(Equilíbrio)']
    
    # VALORES ATUALIZADOS PARA BATER COM O TEXTO DA DISSERTAÇÃO (Modelo de 90.81%)
    valores_classe_0 = [0.91, 0.90, 0.91] # Código Seguro
    valores_classe_1 = [0.90, 0.91, 0.91] # Com Bug
    
    # Configuração da posição das barras
    x = np.arange(len(metricas))
    width = 0.35 # Largura de cada barra
    
    # Configurar o estilo e tamanho do gráfico
    plt.style.use('seaborn-v0_8-whitegrid') # Fundo branco com linhas de grade suaves
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Desenhar as barras (usando cores académicas sóbrias: Azul e Laranja escuro)
    rects1 = ax.bar(x - width/2, valores_classe_0, width, label='Código Seguro (0)', color='#4C72B0')
    rects2 = ax.bar(x + width/2, valores_classe_1, width, label='Com Bug (1)', color='#DD8452')
    
    # Adicionar títulos e rótulos
    ax.set_ylabel('Pontuação (0.0 a 1.0)', fontsize=12, weight='bold')
    ax.set_title('Métricas de Avaliação: Classificador Random Forest', fontsize=14, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metricas, fontsize=12, weight='bold')
    
    # Limitar o eixo Y de 0 até 1.1 para não colar o texto no topo
    ax.set_ylim(0, 1.1)
    
    # Adicionar a legenda
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    
    # Adicionar os números exatos em cima de cada barra
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=11)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=11)
    
    # Ajustar layout e gravar a imagem em alta resolução (300 dpi)
    plt.tight_layout()
    nome_arquivo = 'Figura_Y_Metricas_Desempenho.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    
    print(f"✅ Sucesso! Gráfico de alta resolução salvo como: {nome_arquivo}")

if __name__ == "__main__":
    gerar_grafico_metricas()