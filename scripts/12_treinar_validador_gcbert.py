import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def treinar_modelo_semantico():
    print("🌲 Iniciando o Treinamento do Modelo Semântico (Random Forest)...")
    
    caminho_embeddings = 'C:/Dissertacao/data_bases/graphcodebert/embeddings_graphcodebert.csv'
    
    print("📂 Carregando os vetores salvos...")
    df = pd.read_csv(caminho_embeddings)
    
    # --- O AJUSTE ESTÁ AQUI ---
    # Separando o alvo (y)
    y = df['Tem_Fuga_de_Recurso'].values
    
    # Separando os grupos (ID_Par) - Não entra no treino como dado, apenas como regra de divisão
    grupos = df['ID_Par'].values
    
    # Criando o X: Removemos o alvo E o ID_Par (que é string)
    # Isso garante que o X só tenha as colunas dim_0, dim_1... que são números (floats)
    X = df.drop(columns=['Tem_Fuga_de_Recurso', 'ID_Par']).values

    print("🔀 Dividindo dados por GRUPOS (70/30) para evitar Data Leakage...")
    # Usamos GroupShuffleSplit para que o BUG e o FIX do mesmo caso fiquem no mesmo bloco
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=grupos))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"⚙️ Treinando o classificador com {len(X_train)} exemplos...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    print("\n📊 RESULTADOS FINAIS:")
    y_pred = rf.predict(X_test)
    
    print("="*60)
    print(f"🎯 Acurácia Geral: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("-" * 60)
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 60)
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print("="*60)

if __name__ == "__main__":
    treinar_modelo_semantico()