import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def gerar_matriz_confusao_graphcodebert():
    print("🎨 Gerando a Matriz de Confusão (GraphCodeBERT) para a dissertação...")
    
    # NÚMEROS REAIS DO SEU RESULTADO (Acurácia 82.39%)
    # [[VN, FP], [FN, VP]]
    matriz_confusao = np.array([[170, 43], 
                                [32, 181]])
    
    # Nomes das classes
    labels = ["Código Seguro (0)", "Com Bug (1)"]
    
    # Configuração do estilo acadêmico
    sns.set_theme(style="white")
    plt.figure(figsize=(8, 6))
    
    # Criar o Heatmap
    ax = sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=labels, yticklabels=labels,
                     annot_kws={"size": 16, "weight": "bold"},
                     cbar_kws={'label': 'Número de Casos'})
    
    # Títulos e Eixos
    plt.title('Matriz de Confusão: Validador Semântico GraphCodeBERT', fontsize=15, pad=20, weight='bold')
    plt.ylabel('Rótulo Verdadeiro (Ground Truth)', fontsize=12, weight='bold')
    plt.xlabel('Previsão do Modelo (Acurácia 82.39%)', fontsize=12, weight='bold')
    
    plt.tight_layout()
    
    # Salvar em 300 DPI
    nome_arquivo = 'Matriz_Confusao_GraphCodeBERT.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    
    print(f"✅ Sucesso! Gráfico salvo como: {nome_arquivo}")

if __name__ == "__main__":
    gerar_matriz_confusao_graphcodebert()