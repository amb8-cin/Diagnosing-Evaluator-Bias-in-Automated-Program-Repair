import pandas as pd
import re
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt # Adicionado para conseguir plotar o gráfico

def limpar_codigo_java(codigo):
    if not isinstance(codigo, str):
        return ""
    codigo = re.sub(r'//.*', '', codigo)
    codigo = re.sub(r'/\*.*?\*/', '', codigo, flags=re.DOTALL)
    codigo = re.sub(r'\s+', ' ', codigo)
    return codigo.strip().lower()

def main():
    print("🚀 A iniciar o Treino do Validador (O Juiz)...\n")
    
    # 1. CARREGAR OS DADOS
    print("1. A carregar o dataset de treino filtrado...")
    df_completo = pd.read_csv('./data_bases/treino_validador_final.csv', sep=';')
    print(f"   -> Total de exemplos carregados para treino: {len(df_completo)}")

    # 2. LIMPEZA DE DADOS
    print("\n2. A realizar Limpeza de Código...")
    df_completo['Codigo_Limpo'] = df_completo['Codigo_Snippet'].apply(limpar_codigo_java)
    
    X = df_completo['Codigo_Limpo']
    y = df_completo['Tem_Fuga_de_Recurso']

    # 3. VETORIZAÇÃO (TF-IDF)
    print("\n3. A realizar Vetorização (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=1500) 
    X_vetorizado = vectorizer.fit_transform(X)

    # 4. DIVIDIR DADOS
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_vetorizado, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. TREINAR O MODELO
    print("\n5. A treinar o Modelo (Random Forest)...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_treino, y_treino)

    # 6. AVALIAÇÃO INTERNA DO MODELO E MATRIZ DE CONFUSÃO
    print("\n6. A avaliar o Modelo com os dados de Teste internos...")
    previsoes = modelo.predict(X_teste)
    
    print("\n" + "="*50)
    print("📊 DESEMPENHO INTERNO DO VALIDADOR")
    print("="*50)
    print(f"Precisão Geral (Accuracy): {accuracy_score(y_teste, previsoes) * 100:.2f}%\n")
    print(classification_report(y_teste, previsoes, target_names=["Código Correto (0)", "Com Bug (1)"]))
    
    # --- MATRIZ DE CONFUSÃO (TEXTO E GRÁFICO) ---
    print("\nMatriz de Confusão (Valores Reais):")
    matriz = confusion_matrix(y_teste, previsoes)
    print(matriz)
    print("="*50)

    # 7. GUARDAR O MODELO
    print("\n7. A guardar o modelo e o vetorizador no disco...")
    joblib.dump(modelo, 'validador_resource_leak.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("✅ Sucesso! Ficheiros '.pkl' foram criados.")

    # 8. PLOTAR O GRÁFICO NA TELA
    print("\n🎨 A abrir o gráfico da Matriz de Confusão. Feche a janela para terminar o script.")
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["Correto (0)", "Com Bug (1)"])
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusão do Treino")
    plt.show()

if __name__ == "__main__":
    main()