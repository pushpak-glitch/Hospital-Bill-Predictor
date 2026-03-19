import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


train_data = pd.read_csv("/Users/pushpak/Hospital Bill Prediction/final_train_data.csv")
test_data = pd.read_csv("/Users/pushpak/Hospital Bill Prediction/final_test_data.csv")


train_data_feature = train_data.drop(["Patient_ID", "Bill_Amount"], axis=1)
train_data_label = train_data["Bill_Amount"]

test_data_feature = test_data.drop(["Patient_ID", "Bill_Amount"], axis=1)
test_data_label = test_data["Bill_Amount"]



num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, selector(dtype_include=np.number)),
    ("cat", cat_pipeline, selector(dtype_include=object))
])



models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

rmse_scores = {}



for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipe.fit(train_data_feature, train_data_label)
    label_predict = pipe.predict(test_data_feature)
    
    rmse = np.sqrt(mean_squared_error(test_data_label, label_predict))
    rmse_scores[name] = rmse
    
    print(f"{name} RMSE_score: {rmse:.2f}")
    
best_model = min(rmse_scores, key=rmse_scores.get)

print("Best Model for prediction:", best_model)
print("model Lowest RMSE:", rmse_scores[best_model])