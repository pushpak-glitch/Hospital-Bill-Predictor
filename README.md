# Hospital Bill Predictor

This project predicts **hospital bill amount** for patients using a machine learning pipeline.

## Technologies Used
- Python  
- Pandas (data loading)  
- scikit‑learn (preprocessing + RandomForestRegressor)  
- Joblib (model & pipeline persistence)  

## Dataset
- `final_train_data.csv`: Training data with features and `Bill_Amount` label.  
- `final_test_without_label.csv`: Test data (no labels) for predictions.  

## Model Pipeline
- Numerical features: imputed with median and scaled using `StandardScaler`.  
- Categorical features: imputed with mode and encoded with `OneHotEncoder`.  
- Model: `RandomForestRegressor` (n_estimators=100) chosen based on lowest RMSE.  

## Key Result
- Best model: **RandomForestRegressor**  
- Test RMSE: **≈ 8441**  

## How to Run
1. Place `final_train_data.csv` and `final_test_without_label.csv` in the folder.  
2. Run:  
   ```bash
   python "bill prediction.py"
