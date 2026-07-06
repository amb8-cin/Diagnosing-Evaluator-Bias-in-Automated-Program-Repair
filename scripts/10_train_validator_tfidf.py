import pandas as pd
import re
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt # Added to be able to plot the graph

def clean_java_code(code):
    if not isinstance(code, str):
        return ""
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'\s+', ' ', code)
    return code.strip().lower()

def main():
    print("🚀 Starting the Validator Training (The Judge)...\n")
    
    # 1. LOAD DATA
    print("1. Loading the filtered training dataset...")
    full_df = pd.read_csv('./data_bases/04_final/training_validator_final.csv', sep=';')
    print(f"   -> Total examples loaded for training: {len(full_df)}")

    # 2. DATA CLEANING
    print("\n2. Performing Code Cleaning...")
    full_df['Clean_Code'] = full_df['Code_Snippet'].apply(clean_java_code)
    
    X = full_df['Clean_Code']
    y = full_df['Has_Resource_Leak']

    # 3. VECTORIZATION (TF-IDF)
    print("\n3. Performing Vectorization (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=1500) 
    X_vectorized = vectorizer.fit_transform(X)

    # 4. SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. TRAIN THE MODEL
    print("\n5. Training the Model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. INTERNAL MODEL EVALUATION AND CONFUSION MATRIX
    print("\n6. Evaluating the Model with internal Test data...")
    predictions = model.predict(X_test)
    
    print("\n" + "="*50)
    print("📊 VALIDATOR INTERNAL PERFORMANCE")
    print("="*50)
    print(f"Overall Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
    print(classification_report(y_test, predictions, target_names=["Correct Code (0)", "With Bug (1)"]))
    
    # --- CONFUSION MATRIX (TEXT AND PLOT) ---
    print("\nConfusion Matrix (Actual Values):")
    matrix = confusion_matrix(y_test, predictions)
    print(matrix)
    print("="*50)

    # 7. SAVE THE MODEL
    print("\n7. Saving the model and vectorizer to disk...")
    joblib.dump(model, 'validator_resource_leak.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("✅ Success! '.pkl' files have been created.")

    # 8. PLOT THE GRAPH ON SCREEN
    print("\n🎨 Opening the Confusion Matrix plot. Close the window to terminate the script.")
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Correct (0)", "With Bug (1)"])
    disp.plot(cmap='Blues')
    plt.title("Training Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()