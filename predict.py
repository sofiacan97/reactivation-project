import pandas as pd
import os
from src.utils import PATH_BUNDLE, PATH_NEW_DATA, PATH_OUTPUT_EXCEL_PREDICTION
from src.model import load_model_bundle
from src.scoring import apply_model_to_new_data



def run_prediction():

    # --- 1. BUNDLE UPLOAD ---
    print(f"Model from {PATH_BUNDLE}...")
    bundle = load_model_bundle(PATH_BUNDLE)


    # --- 2. NEW DATA SCORING ---
    print(f"Data: {PATH_NEW_DATA}...")
    new_data_original = pd.read_csv(PATH_NEW_DATA)
    
    cols_to_drop = ["customer_id", "segment_P12M", "reactivation_day", "reactivation", "filter_date"]
    new_data_model = new_data_original.drop(columns=[c for c in cols_to_drop if c in new_data_original.columns],
                                            errors="ignore")
    new_data_model["last_two_purchase_diff"] = pd.to_numeric(new_data_model["last_two_purchase_diff"],
                                                             errors="coerce")

    print("Predictions...")
    results = apply_model_to_new_data(
        model=bundle['model'],
        df_new=new_data_model,
        scaler=bundle['scaler'],
        encoder=bundle['encoder'],
        num_cols=bundle['num_cols'],
        cat_cols=bundle['cat_cols'],
        threshold=0.55 
    )

    # --- 3. EXPORT ---
    new_data_original["prediction"] = results["prediction"].values
    new_data_original["probability"] = results["probability"].values
    
    new_data_original = new_data_original.sort_values(by="probability", ascending=False)

    os.makedirs('outputs/reports', exist_ok=True)
    new_data_original.to_csv(PATH_OUTPUT_EXCEL_PREDICTION, index=False)
    
    print(f"Success! {len(new_data_original)} customer processed.")
    print(f"File saved in: {PATH_OUTPUT_EXCEL_PREDICTION}")

if __name__ == "__main__":
    run_prediction()