import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_train_data(X_train: pd.DataFrame):
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

    X_train = X_train.copy()
    X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].median())
    X_train[cat_cols] = X_train[cat_cols].fillna("Missing")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(X_train[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=X_train.index)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_train[num_cols])
    scaled_df = pd.DataFrame(scaled, columns=num_cols, index=X_train.index)

    return pd.concat([scaled_df, encoded_df], axis=1), scaler, encoder, num_cols, cat_cols

def preprocess_transform_data(df, scaler, encoder, num_cols, cat_cols):
    df = df.copy()

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Missing")

    scaled_df = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols, index=df.index)
    encoded_df = pd.DataFrame(encoder.transform(df[cat_cols]), 
                              columns=encoder.get_feature_names_out(cat_cols), index=df.index)

    return pd.concat([scaled_df, encoded_df], axis=1)