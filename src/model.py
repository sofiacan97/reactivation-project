import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from .data_processing import preprocess_train_data, preprocess_transform_data

def train_final_decision_tree(X, y, max_depth=5, min_samples_leaf=5, random_state=42):

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    X_train, scaler, encoder, num_cols, cat_cols = preprocess_train_data(X_train_raw)
    
    X_test = preprocess_transform_data(X_test_raw, scaler, encoder, num_cols, cat_cols)

    dt = DecisionTreeClassifier(
        max_depth=max_depth, 
        min_samples_leaf=min_samples_leaf, 
        random_state=random_state, 
        class_weight="balanced"
    )
    dt.fit(X_train, y_train)

    print(f"Accuracy Train: {dt.score(X_train, y_train):.3f}")
    
    return dt, scaler, encoder, num_cols, cat_cols, X_test, y_test

def save_model_bundle(model, scaler, encoder, num_cols, cat_cols, folder_path='outputs/model'):

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    bundle = {
        'model': model,
        'scaler': scaler,
        'encoder': encoder,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }
    
    file_path = os.path.join(folder_path, 'reactivation_bundle_v2.joblib')
    joblib.dump(bundle, file_path)
    print(f"Bundle saved: {file_path}")

def load_model_bundle(file_path='outputs/model/reactivation_bundle.joblib'):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No bundle found {file_path}")
    return joblib.load(file_path)