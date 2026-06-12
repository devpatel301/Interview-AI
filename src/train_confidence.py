import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_model():
    data_path = "data/kaggle_interview_dataset.csv"
    if not os.path.exists(data_path):
        print("Dataset not found. Run generate_kaggle_mock.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Features and Target
    X = df[["eye_contact_avg", "posture_avg", "speaking_pace", "pause_ratio"]]
    y = df["confidence"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training Random Forest Regressor for Confidence Scoring...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print("\n--- Model Evaluation Stats ---")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error:  {mse:.4f}")
    print(f"R2 Score:            {r2:.4f}")
    
    print("\n--- Feature Importances ---")
    importances = model.feature_importances_
    for feat, imp in zip(X.columns, importances):
        print(f"{feat:20s}: {imp:.4f}")
        
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/confidence_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_model()
