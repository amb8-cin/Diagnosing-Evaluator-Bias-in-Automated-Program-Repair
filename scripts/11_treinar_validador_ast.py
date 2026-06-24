import pandas as pd
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    print("🚀 Iniciando o Treinamento do Validador Estrutural (AST)...\n")
    
    # 1. CARREGAR OS DADOS
    # Vamos usar o conjunto de treino que separamos no split.py (as 1349 linhas)
    caminho_csv = 'C:/Dissertacao/data_bases/ast/dataset_ast_treino.csv'
    print(f"1. Carregando o dataset de treino: {caminho_csv}...")
    
    try:
        df = pd.read_csv(caminho_csv)
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo não encontrado em {caminho_csv}")
        return

    print(f"   -> Total de exemplos carregados: {len(df)}")

    # Lidar com possíveis valores nulos caso algum código não tenha extraído features
    df['AST_Features'] = df['AST_Features'].fillna("SEM_FEATURES")

    X = df['AST_Features']
    y = df['Tem_Fuga_de_Recurso']

    # 2. VETORIZAÇÃO (Code Embeddings)
    # O Vectorizer agora vai contar as "folhas" da árvore (ex: TryStatement, MethodCall_release)
    print("\n2. Realizando Vetorização Estrutural...")
    vectorizer = TfidfVectorizer() 
    X_vetorizado = vectorizer.fit_transform(X)
    
    print(f"   -> Vocabulário estrutural mapeado: {len(vectorizer.get_feature_names_out())} nós sintáticos.")

    # 3. DIVIDIR DADOS (Validação Interna de 80/20)
    # Conforme escrito na metodologia, separamos 80% do treino para ensinar e 20% para testar a precisão interna
    print("\n3. Separando 80% para treino e 20% para validação interna...")
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X_vetorizado, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. TREINAR O MODELO
    print("\n4. Treinando o Modelo (Random Forest)...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_treino, y_treino)

    # 5. AVALIAÇÃO INTERNA DO MODELO E MATRIZ DE CONFUSÃO
    print("\n5. Avaliando o Modelo Estrutural com os 20% de teste interno...")
    previsoes = modelo.predict(X_teste)
    
    acuracia = accuracy_score(y_teste, previsoes) * 100
    print("\n" + "="*50)
    print("📊 DESEMPENHO INTERNO DO VALIDADOR AST")
    print("="*50)
    print(f"Precisão Global (Accuracy): {acuracia:.2f}%\n")
    print(classification_report(y_teste, previsoes, target_names=["Corrigido (0)", "Com Bug (1)"]))
    
    # --- MATRIZ DE CONFUSÃO ---
    print("\nMatriz de Confusão (Valores Reais):")
    matriz = confusion_matrix(y_teste, previsoes)
    print(matriz)
    print("="*50)

    # 6. GUARDAR O MODELO
    print("\n6. Salvando o oráculo AST no disco...")
    # Salvamos com nomes diferentes para não sobrescrever os modelos antigos do TF-IDF
    joblib.dump(modelo, 'C:/Dissertacao/trained_models/validador_ast_rf.pkl')
    joblib.dump(vectorizer, 'C:/Dissertacao/trained_models/vectorizer_ast.pkl')
    print("✅ Sucesso! Arquivos 'validador_ast_rf.pkl' e 'vectorizer_ast.pkl' foram criados.")

    # 7. PLOTAR O GRÁFICO NA TELA
    print("\n🎨 Abrindo o gráfico da Matriz de Confusão. Feche a janela para encerrar.")
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["Corrigido (0)", "Com Bug (1)"])
    disp.plot(cmap='Greens') # Mudamos a cor para Verde para não confundir com o gráfico Azul do Baseline!
    plt.title("Matriz de Confusão do Treino (AST)")
    plt.show()

if __name__ == "__main__":
    main()