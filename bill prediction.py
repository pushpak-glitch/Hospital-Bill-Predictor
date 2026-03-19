import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


if not os.path.exists(PIPELINE_FILE):
    # Load training data
    df = pd.read_csv("/Users/pushpak/Hospital Bill Prediction/final_train_data.csv")

    y = df["Bill_Amount"]
    x = df.drop(["Patient_ID", "Bill_Amount"],axis=1)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, selector(dtype_include="number")),
        ("cat", cat_pipeline, selector(dtype_include="object"))
    ])

    X_prepared = preprocessor.fit_transform(x)

    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_prepared, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(preprocessor, PIPELINE_FILE)
    print("Model trained and pipeline saved.")

else:
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("/Users/pushpak/Hospital Bill Prediction/final_test_without_label.csv")
    input_prepared = preprocessor.transform(input_data)

    predictions = model.predict(input_prepared)
    input_data["prediction"] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Prediction done. output.csv saved")
