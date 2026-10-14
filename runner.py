from src.utils import PATH_RAW_TRAIN
import pandas as pd
from src.model import train_final_decision_tree, save_model_bundle
from src.scoring import evaluate_model

def run_full_pipeline():
    # --- 1. UPLOAD ---
    print("upload data...")
    df = pd.read_csv(PATH_RAW_TRAIN)
    
    cols_to_drop = ["customer_id", "segment_P12M", "reactivation_day", "reactivation", "filter_date"]

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df["reactivation"]

    X["last_two_purchase_diff"] = pd.to_numeric(X["last_two_purchase_diff"], errors="coerce")

    # --- 2. TRAINING ---
    print("training...")
    model, scaler, encoder, num_cols, cat_cols, X_test, y_test = train_final_decision_tree(X, y)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, y_prob_test, threshold=0.55)

    # --- 3. SAVING ---

    save_model_bundle(model, scaler, encoder, num_cols, cat_cols)

if __name__ == "__main__":
    run_full_pipeline()