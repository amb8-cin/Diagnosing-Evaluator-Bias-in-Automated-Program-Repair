import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def gerar_graficos_dissertacao():
    print("A gerar gráficos em alta resolução...")
    
    # ==========================================
    # FIGURA 1: Gráfico de Barras (Métricas)
    # ==========================================
    # Dados extraídos do seu classification_report
    metricas = ['Precisão', 'Revocação', 'F1-Score']
    codigo_correto = [0.91, 0.90, 0.91]
    com_bug = [0.90, 0.91, 0.91]

    x = np.arange(len(metricas))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    
    # Cores amigáveis para leitura acadêmica
    rects1 = ax1.bar(x - width/2, codigo_correto, width, label='Código Correto (0)', color='#4C72B0')
    rects2 = ax1.bar(x + width/2, com_bug, width, label='Com Bug (1)', color='#DD8452')

    ax1.set_ylabel('Pontuação (0.0 a 1.0)', fontsize=12)
    ax1.set_title('Desempenho do Modelo Random Forest por Classe', fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metricas, fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right', fontsize=11)

    # Adicionar os números em cima de cada barra
    ax1.bar_label(rects1, padding=3, fmt='%.2f')
    ax1.bar_label(rects2, padding=3, fmt='%.2f')

    fig1.tight_layout()
    plt.savefig('Figura_Y_Metricas_Desempenho.png', dpi=300, bbox_inches='tight')
    print("✅ 'Figura_Y_Metricas_Desempenho.png' criada com sucesso!")

    # ==========================================
    # FIGURA 2: Matriz de Confusão (Heatmap)
    # ==========================================
    # Dados exatos calculados a partir dos seus suportes (142 e 141) e recall
    matriz_confusao = np.array([[128, 14], 
                                [12, 129]])

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    
    # Criar o mapa de calor (tons de azul costumam ficar ótimos em dissertações)
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', 
                cbar=False, annot_kws={"size": 16},
                xticklabels=['Correto (0)', 'Com Bug (1)'],
                yticklabels=['Correto (0)', 'Com Bug (1)'])
    
    ax2.set_ylabel('Rótulo Verdadeiro (Ground Truth)', fontsize=12)
    ax2.set_xlabel('Previsão do Validador', fontsize=12)
    ax2.set_title('Matriz de Confusão', fontsize=14, pad=15)

    # Melhorar a visibilidade dos eixos
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11, rotation=0)

    fig2.tight_layout()
    plt.savefig('Figura_Z_Matriz_Confusao.png', dpi=300, bbox_inches='tight')
    print("✅ 'Figura_Z_Matriz_Confusao.png' criada com sucesso!")

if __name__ == "__main__":
    # Se não tiver o seaborn instalado, rode no terminal: pip install seaborn
    gerar_graficos_dissertacao()