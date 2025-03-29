import warnings

import matplotlib.pyplot as plt
import pandas as pd
import strategy.betting_logic as betting_logic
from data.data_prep import DataPrep
from data.data_preprocessing import Preprocessor
from features.build_features import FeaturePipeline
from model.train import train_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.simplefilter(action="ignore", category=FutureWarning)


def plot_pnl(df):
    """Plots cumulative profit and loss over time."""
    plot_df = df[df["unit_pnl"] != 0]
    plt.plot(plot_df["unit_pnl"].cumsum(), label="Cumulative PnL")
    plt.xlabel("Games")
    plt.ylabel("Profit/Loss (units)")
    plt.title("Betting Strategy Performance")
    plt.legend()
    plt.show()


def cross_validate(
    odds_df, X, y, betting_fnc=betting_logic.simple_percentage, init_train_yrs=5
):
    """Performs rolling cross-validation using past seasons to predict future ones."""
    # Have to redo the below with new indices
    cv_year_indices = X.groupby("season").head(1).index
    cv_year_indices = cv_year_indices[init_train_yrs:].append(pd.Index([df.index[-1]]))

    for i in range(len(cv_year_indices) - 1):
        train_idx, test_idx = cv_year_indices[i], cv_year_indices[i + 1]

        # Train-test split
        X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
        X_test = X.iloc[train_idx:test_idx]

        # Train model & predict
        model = train_model(X_train, y_train)
        odds_df.loc[train_idx : test_idx - 1, "pred"] = model.predict(X_test)

    # Apply betting logic & plot results
    odds_df = betting_fnc(odds_df)
    # TODO: model.to_json (name)
    return model, odds_df


def evaluate_cv(model, df):
    plot_pnl(df)
    y_pred = df["pred"]
    y_test = df["total"]

    # TODO: Brier score?
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    if model.__class__.__name__ == "Booster":
        # Get the most predictive features
        feature_importances = model.feature_importance(importance_type="gain")
        feature_names = model.feature_name()
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        importance_df["Percentage"] = (
            importance_df["Importance"] / importance_df["Importance"].sum()
        ) * 100
        top_n_features = importance_df.sort_values(
            by="Importance", ascending=False
        ).head(3)
        print(top_n_features[["Feature", "Percentage"]])
    print("===============================")
    print(f"Model Performance ({model.__class__.__name__}):")
    print("R2 Score:\n", r2)
    print("MSE:\n", mse)
    print("MAE:\n", mae)


if __name__ == "__main__":
    # python src/cfb/backtest.py
    print("Step 1: Loading data...")
    data_prep = DataPrep(dataset="cfb")
    raw_data = data_prep.get_data()

    print("Step 2: Preprocessing data...")
    preprocessor = Preprocessor(raw_data, "total")
    odds_df, X, y = preprocessor.preprocess_data()

    print("Step 3: Feature engineering...")
    feature_pipeline = FeaturePipeline(X)
    X = feature_pipeline.engineer_features()

    # 4. Train the Model
    print("Step 4: Training and evaluating the model...")
    model, df = cross_validate(odds_df, X, y)
    breakpoint()
    # 5. Evaluate the Model
    print("Step 5: Evaluating the model...")
    evaluate_cv(model, df)
