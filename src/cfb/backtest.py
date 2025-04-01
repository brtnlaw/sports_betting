import datetime as dt
import os
import pickle as pkl
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import strategy.betting_logic as betting_logic
from data.data_prep import DataPrep

# TODO: FIX
from model.train import train_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_pkl_if_exists(
    name_str, betting_fnc=betting_logic.simple_percentage, file_type="df"
):
    """Helper function to load 'model' or 'df' from a str."""
    assert file_type in ["model", "df"], "Pick a file_type in 'model', 'df'"
    if file_type == "model":
        file_path = f"src/cfb/model/models/{name_str}.mdl"
        if not os.path.exists(file_path):
            raise Exception(f"No properly configured {file_type}.")
        result = lgb.Booster(model_file=file_path)
    else:
        file_path = f"src/cfb/model/models/{name_str}_{betting_fnc.__name__}.pkl"
        if not os.path.exists(file_path):
            raise Exception(f"No properly configured {file_type}.")
        with open(file_path, "rb") as file:
            result = pkl.load(file)
    return result


def plot_pnl(model_str, betting_fnc=betting_logic.simple_percentage):
    model_df = load_pkl_if_exists(model_str, betting_fnc, "df")
    model_df.fillna(0, inplace=True)
    plot_model_df = model_df[model_df["unit_pnl"] != 0]
    plot_model_df.reset_index(drop=True, inplace=True)

    plt.plot(plot_model_df["unit_pnl"].cumsum(), label=model_str)
    plt.xlabel("Games")
    plt.ylabel("Profit/Loss (units)")
    plt.title("Model Betting Strategy Performance")
    plt.legend()
    plt.show()


def plot_pnl_comparison(
    model_str, baseline_str="model_3_29_25", betting_fnc=betting_logic.simple_percentage
):
    # TODO: Include target in model name.
    plot_model_df = load_pkl_if_exists(model_str, betting_fnc, "df")
    plot_baseline_df = load_pkl_if_exists(baseline_str, betting_fnc, "df")

    plot_model_df.fillna(0, inplace=True)
    plot_model_df = plot_model_df[plot_model_df["unit_pnl"] != 0]
    plot_model_df.reset_index(drop=True, inplace=True)
    plot_baseline_df.fillna(0, inplace=True)
    plot_baseline_df = plot_baseline_df[plot_baseline_df["unit_pnl"] != 0]
    plot_baseline_df.reset_index(drop=True, inplace=True)

    plt.plot(
        plot_model_df["unit_pnl"].cumsum(), label=f"{model_str}_{betting_fnc.__name__}"
    )
    plt.plot(
        plot_baseline_df["unit_pnl"].cumsum(),
        label=f"{baseline_str}_{betting_fnc.__name__}",
    )
    plt.xlabel("Games")
    plt.ylabel("Profit/Loss (units)")
    plt.title("Betting Strategy Performance")
    plt.legend()
    plt.show()


def cross_validate(
    odds_df,
    X,
    y,
    file_name=None,
    betting_fnc=betting_logic.simple_percentage,
    init_train_yrs=5,
):
    """Performs rolling cross-validation using past seasons to predict future ones."""
    cv_year_indices = X.groupby("season").head(1).iloc[init_train_yrs:].index
    cv_year_spots = [X.index.get_loc(idx) for idx in cv_year_indices] + [len(X)]

    for i in range(len(cv_year_spots) - 1):
        train_idx, test_idx = cv_year_spots[i], cv_year_spots[i + 1]

        # Train-test split
        X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
        X_test = X.iloc[train_idx:test_idx]

        # Train model & predict
        model = train_model(X_train, y_train)
        preds = model.predict(X_test)
        odds_df.iloc[train_idx:test_idx, odds_df.columns.get_loc("pred")] = preds

    # Apply betting logic & plot results
    odds_df = betting_fnc(odds_df)
    if not file_name:
        td = dt.datetime.today()
        file_name = f"model_{td.month}_{td.day}_{td.year%100}"
    if model.__class__.__name__ == "Booster":
        model.save_model(f"src/cfb/model/models/{file_name}.mdl")
    with open(
        f"src/cfb/model/models/{file_name}_{betting_fnc.__name__}.pkl", "wb"
    ) as f:
        pkl.dump(odds_df, f)
    return model, odds_df


def model_metrics(
    model_str,
    baseline_str="baseline_3_30_25",
    betting_fnc=betting_logic.simple_percentage,
):
    # NOTE: Max drawdown, Brier score
    plot_pnl_comparison(model_str, baseline_str)

    model = load_pkl_if_exists(model_str, betting_fnc, "model")
    baseline = load_pkl_if_exists(baseline_str, betting_fnc, "model")
    model_df = load_pkl_if_exists(model_str, betting_fnc, "df")
    baseline_df = load_pkl_if_exists(baseline_str, betting_fnc, "df")

    str_list = [model_str, baseline_str]
    df_list = [model_df, baseline_df]
    model_list = [model, baseline]

    print("==============================================")
    print(
        f"PNL delta from the model: {(model_df["unit_pnl"].sum() - baseline_df["unit_pnl"].sum()).round(2)} units"
    )

    for i in range(2):
        df_list[i].reset_index(drop=True, inplace=True)
        df_list[i].dropna(inplace=True)
        print("==============================================")
        print(f"Model Performance ({model.__class__.__name__}): {str_list[i]}")
        y_pred = df_list[i]["pred"]
        y_test = df_list[i]["total"]
        r2 = r2_score(y_test, y_pred)
        print("R2 Score:\n", r2)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE:\n", mse)
        mae = mean_absolute_error(y_test, y_pred)
        print("MAE:\n", mae)
        if model_list[i].__class__.__name__ == "Booster":
            # Get the most predictive features
            feature_importances = model_list[i].feature_importance(
                importance_type="gain"
            )
            feature_names = model_list[i].feature_name()
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

    # 5. Evaluate the Model
    print("Step 5: Evaluating the model...")
    model_metrics(model, df)


"""
df = pipeline[-1].fit_transform(raw_data)
df.columns = df.columns.str.split('__', n=1).str[-1]"
# Workflow is get raw_data, have one step to get the odds_df out (with prediction, etc.), then plug the whole thing into pipeline
"""
