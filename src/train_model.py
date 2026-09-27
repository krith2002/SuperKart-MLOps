import joblib
from math import sqrt
import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from huggingface_hub import HfApi, hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "models"))
TARGET_COL = "Product_Store_Sales_Total"
DATASET_REPO_ID = "Krithika2002/superkart-sales-data"
MODEL_REPO_ID = "Krithika2002/superkart-sales-model"


def load_data():
    train_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="train.csv",
        repo_type="dataset",
    )
    test_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="test.csv",
        repo_type="dataset",
    )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def prepare_features(df):
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    y = df[TARGET_COL]
    X = pd.get_dummies(X, drop_first=True)
    return X, y


def train_and_evaluate():
    train_df, test_df = load_data()

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    # Align columns so train and test are consistent
    X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

    parameter_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 15],
        "min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        parameter_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print(f"Best parameters: {search.best_params_}")
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set; skipping model registration.")
        return

    HfApi(token=token).upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="best_model.pkl",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Register tuned SuperKart sales model",
    )
    print(f"Registered best_model.pkl in {MODEL_REPO_ID}.")


def main():
    train_and_evaluate()


if __name__ == "__main__":
    main()
