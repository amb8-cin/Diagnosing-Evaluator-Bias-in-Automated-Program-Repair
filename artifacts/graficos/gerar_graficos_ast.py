import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuração de estilo elegante para trabalhos acadêmicos
sns.set_theme(style="whitegrid")

def plotar_metricas_ast():
    # Dados extraídos do seu log de treinamento AST
    metricas = ['Precision', 'Recall', 'F1-Score']
    corrigido = [0.91, 0.87, 0.89]
    com_bug = [0.87, 0.91, 0.89]

    x = np.arange(len(metricas))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Cores: Verde para Corrigido, Vermelho para Bug (facilita a leitura da banca)
    rects1 = ax.bar(x - width/2, corrigido, width, label='Corrigido (0)', color='#2ca02c')
    rects2 = ax.bar(x + width/2, com_bug, width, label='Com Bug (1)', color='#d62728')

    ax.set_ylabel('Pontuação', fontsize=12)
    ax.set_title('Desempenho do Validador Estrutural (AST)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metricas, fontsize=11)
    ax.set_ylim(0, 1.1) # Vai até 1.1 para sobrar espaço para a legenda
    ax.legend(loc='lower right', fontsize=11)

    # Adiciona os números em cima de cada barra
    ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=11)
    ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=11)

    plt.tight_layout()
    plt.savefig('Figura_Y_Metricas_Desempenho_AST.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico de métricas salvo como 'Figura_Y_Metricas_Desempenho_AST.png'")

def plotar_matriz_confusao_ast():
    # Valores reais da Matriz de Confusão do seu log
    # [[VN, FP], [FN, VP]] -> [[117, 18], [12, 123]]
    matriz = np.array([[117, 18], [12, 123]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Usando tons de verde ('Greens') para diferenciar visualmente do TF-IDF que era azul
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Corrigido (0)', 'Com Bug (1)'],
                yticklabels=['Corrigido (0)', 'Com Bug (1)'],
                annot_kws={"size": 16, "fontweight": "bold"})
    
    plt.ylabel('Rótulo Verdadeiro', fontsize=12, fontweight='bold')
    plt.xlabel('Previsão do Oráculo (AST)', fontsize=12, fontweight='bold')
    plt.title('Matriz de Confusão - Modelo AST', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('Figura_Z_Matriz_Confusao_AST.png', dpi=300, bbox_inches='tight')
    print("✅ Matriz de confusão salva como 'Figura_Z_Matriz_Confusao_AST.png'")

if __name__ == "__main__":
    print("📊 Gerando imagens de alta resolução para o LaTeX...")
    plotar_metricas_ast()
    plotar_matriz_confusao_ast()
    print("🎉 Gráficos gerados com sucesso na sua pasta!")