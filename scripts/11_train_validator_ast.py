import pandas as pd
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    print("🚀 Starting Structural Validator (AST) Training...\n")
    
    # 1. LOAD DATA
    # We will use the training set we separated in split.py (the 1349 lines)
    csv_path = 'C:/Dissertacao/data_bases/ast/dataset_ast_train.csv'
    print(f"1. Loading the training dataset: {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {csv_path}")
        return

    print(f"   -> Total examples loaded: {len(df)}")

    # Handle possible null values in case some code did not extract features
    df['AST_Features'] = df['AST_Features'].fillna("NO_FEATURES")

    X = df['AST_Features']
    y = df['Has_Resource_Leak']

    # 2. VECTORIZATION (Code Embeddings)
    # The Vectorizer will now count the "leaves" of the tree (e.g., TryStatement, MethodCall_release)
    print("\n2. Performing Structural Vectorization...")
    vectorizer = TfidfVectorizer() 
    X_vectorized = vectorizer.fit_transform(X)
    
    print(f"   -> Mapped structural vocabulary: {len(vectorizer.get_feature_names_out())} syntactic nodes.")

    # 3. SPLIT DATA (80/20 Internal Validation)
    # As described in the methodology, we separate 80% of the training data to teach and 20% to test internal accuracy
    print("\n3. Separating 80% for training and 20% for internal validation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. TRAIN THE MODEL
    print("\n4. Training the Model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. INTERNAL MODEL EVALUATION AND CONFUSION MATRIX
    print("\n5. Evaluating the Structural Model with the 20% internal test data...")
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions) * 100
    print("\n" + "="*50)
    print("📊 INTERNAL PERFORMANCE OF AST VALIDATOR")
    print("="*50)
    print(f"Overall Accuracy: {accuracy:.2f}%\n")
    print(classification_report(y_test, predictions, target_names=["Fixed (0)", "With Bug (1)"]))
    
    # --- CONFUSION MATRIX ---
    print("\nConfusion Matrix (Actual Values):")
    confusion_matrix_data = confusion_matrix(y_test, predictions)
    print(confusion_matrix_data)
    print("="*50)

    # 6. SAVE THE MODEL
    print("\n6. Saving the AST oracle to disk...")
    # We save with different names to avoid overwriting old TF-IDF models
    joblib.dump(model, 'C:/Dissertacao/trained_models/validador_ast_rf.pkl')
    joblib.dump(vectorizer, 'C:/Dissertacao/trained_models/vectorizer_ast.pkl')
    print("✅ Success! Files 'validador_ast_rf.pkl' and 'vectorizer_ast.pkl' were created.")

    # 7. PLOT THE GRAPH ON SCREEN
    print("\n🎨 Opening the Confusion Matrix graph. Close the window to exit.")
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_data, display_labels=["Fixed (0)", "With Bug (1)"])
    display.plot(cmap='Greens') # Changed color to Green to avoid confusion with the Blue Baseline graph!
    plt.title("Confusion Matrix of Training (AST)")
    plt.show()

if __name__ == "__main__":
    main()