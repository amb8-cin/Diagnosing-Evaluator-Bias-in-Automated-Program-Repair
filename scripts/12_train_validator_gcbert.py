import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_semantic_model():
    print("🌲 Starting Semantic Model Training (Random Forest)...")
    
    embeddings_path = 'C:/Dissertation/data_bases/graphcodebert/embeddings_graphcodebert.csv'
    
    print("📂 Loading saved vectors...")
    df = pd.read_csv(embeddings_path)
    
    # --- THE ADJUSTMENT IS HERE ---
    # Separating the target (y)
    y = df['Has_Resource_Leak'].values
    
    # Separating groups (Pair_ID) - Not used in training as data, only as a split rule
    groups = df['Pair_ID'].values
    
    # Creating X: We remove the target AND the Pair_ID (which is a string)
    # This ensures that X only contains columns dim_0, dim_1... which are numbers (floats)
    X = df.drop(columns=['Has_Resource_Leak', 'Pair_ID']).values

    print("🔀 Splitting data by GROUPS (70/30) to avoid Data Leakage...")
    # We use GroupShuffleSplit so that the BUG and FIX of the same case remain in the same block
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"⚙️ Training the classifier with {len(X_train)} examples...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    print("\n📊 FINAL RESULTS:")
    y_pred = rf.predict(X_test)
    
    print("="*60)
    print(f"🎯 Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("-" * 60)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("="*60)

if __name__ == "__main__":
    train_semantic_model()