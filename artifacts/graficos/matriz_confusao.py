import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def gerar_grafico_dissertacao():
    print("🎨 A gerar o gráfico da Matriz de Confusão para a dissertação...")
    
    # NÚMEROS ATUALIZADOS PARA O MODELO DE 90.81%
    matriz_confusao = np.array([[128, 14], 
                                [12, 129]])
    
    # Nomes das classes
    labels = ["Código Seguro (0)", "Com Bug (1)"]
    
    # Configuração do estilo (fundo branco, ideal para LaTeX/Word)
    sns.set_theme(style="white")
    plt.figure(figsize=(8, 6))
    
    # Criar o Heatmap (Gráfico de calor)
    ax = sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=labels, yticklabels=labels,
                     annot_kws={"size": 16, "weight": "bold"}, # Números grandes e em negrito
                     cbar_kws={'label': 'Número de Casos'})
    
    # Títulos e Eixos com fontes acadêmicas
    plt.title('Matriz de Confusão: Validador de Resource Leaks', fontsize=16, pad=20, weight='bold')
    plt.ylabel('Rótulo Verdadeiro (Ground Truth)', fontsize=14, weight='bold')
    plt.xlabel('Previsão do Modelo', fontsize=14, weight='bold')
    
    # Ajustar as margens para não cortar nenhum texto
    plt.tight_layout()
    
    # Salvar a imagem em altíssima resolução (300 dpi é exigência de revistas científicas)
    nome_arquivo = 'Figura_Z_Matriz_Confusao.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    
    print(f"✅ Sucesso! Gráfico de alta resolução salvo como: {nome_arquivo}")

if __name__ == "__main__":
    gerar_grafico_dissertacao()