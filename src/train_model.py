import joblib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
TARGET_COL = "Product_Store_Sales_Total"


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
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

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "best_model.pkl")
    print(f"Model saved to: {MODEL_DIR / 'best_model.pkl'}")


def main():
    train_and_evaluate()


if __name__ == "__main__":
    main()
