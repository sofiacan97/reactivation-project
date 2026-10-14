import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from .data_processing import preprocess_transform_data

def apply_model_to_new_data(model, df_new, scaler, encoder, num_cols, cat_cols, threshold=0.55):
    X_new_processed = preprocess_transform_data(df_new, scaler, encoder, num_cols, cat_cols)
    y_prob = model.predict_proba(X_new_processed)[:, 1]
    y_pred = np.where(y_prob >= threshold, "yes", "no")
    
    results = df_new.copy()
    results["prediction"] = y_pred
    results["probability"] = y_prob
    return results

def evaluate_model(y_true, y_prob, threshold=0.55):
    y_pred = np.where(y_prob >= threshold, "yes", "no")
    print("\n--- Model Evaluation ---")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))