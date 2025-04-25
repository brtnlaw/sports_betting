import os
from typing import Tuple, Union

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.getenv("PROJECT_ROOT", os.getcwd())
pd.set_option("future.no_silent_downcasting", True)


def load_pkl_if_exists(
    name_str: str,
    target_str: str = "total",
    betting_fnc: str = "spread_probs",
    file_type: str = "odds_df",
) -> Union[Pipeline, pd.DataFrame]:
    """
    Helper function to load 'pipeline', 'contrib_df', or 'odds_df' from a str.

    Args:
        name_str (str): Prefix of file string.
        target_str (str, optional): For model, the target column. Defaults to "total".
        betting_fnc (str, optional): Function to determine bets. Defaults to "spread_probs".
        file_type (str, optional): Type of file to retrieve. Defaults to "df".

    Raises:
        Exception: Necessarily is a 'pipeline' or 'df' file.

    Returns:
        Any[Pipeline, pd.DataFrame]: Returns either a pipeline or DataFrame.
    """
    assert file_type in [
        "pipeline",
        "contrib_df",
        "odds_df",
    ], "Pick a file_type in 'pipeline', 'contrib_df', 'odds_df'"
    if file_type == "pipeline":
        file_path = os.path.join(
            PROJECT_ROOT, f"src/cfb/models/{name_str}_{target_str}_pipeline.pkl"
        )
    elif file_type == "contrib_df":
        file_path = os.path.join(
            PROJECT_ROOT,
            f"src/cfb/models/{name_str}_{target_str}_contrib.pkl",
        )
    elif file_type == "odds_df":
        file_path = os.path.join(
            PROJECT_ROOT,
            f"src/cfb/models/{name_str}_{target_str}_{betting_fnc}.pkl",
        )
    if not os.path.exists(file_path):
        raise Exception(f"No properly configured {file_type} file.")
    result = joblib.load(file_path)
    return result


def get_group_features(
    model_str: str,
    target_str: str = "home_away_spread",
    betting_fnc: str = "spread_probs",
) -> pd.DataFrame:
    """
    Generates a contrib_df grouped by offense and defensive totals

    Args:
        model_str (str): String of the desired model.
        target_str (str, optional): For model, the target column. Defaults to "home_away_spread".
        betting_fnc (str, optional): Function to determine bets. Defaults to "spread_probs".

    Returns:
        pd.DataFrame: Contrib_df with grouped metrics.
    """
    offense_roots = [
        "points_for",
        "third_down_attempts",
        "third_down_successes",
        "fourth_down_attempts",
        "fourth_down_successes",
        "passing_yds_for",
        "ints_thrown",
        "rushing_yds_for",
        "passing_tds",
        "rushing_tds",
    ]
    defense_roots = ["points_against", "passing_yds_given_up"]

    model_contrib_df = load_pkl_if_exists(
        model_str, target_str, betting_fnc, file_type="contrib_df"
    )
    model_contrib_df["offense_total"] = model_contrib_df[
        [
            col
            for col in model_contrib_df.columns
            if any(col.endswith(root) for root in offense_roots)
        ]
    ].sum(axis=1)
    model_contrib_df["defense_total"] = model_contrib_df[
        [
            col
            for col in model_contrib_df.columns
            if any(col.endswith(root) for root in defense_roots)
        ]
    ].sum(axis=1)
    return model_contrib_df


def plot_pnl(
    model_str: str,
    target_str: str = "total",
    betting_fnc: str = "spread_probs",
):
    """
    Plots the net unit pnl given a model.

    Args:
        model_str (str): Model prefix.
        target_str (str, optional): For model, the target column. Defaults to "total".
        betting_fnc (str, optional): Function to determine bets. Defaults to "spread_probs".
    """
    model_df = load_pkl_if_exists(model_str, target_str, betting_fnc, "odds_df")
    model_df.fillna(0, inplace=True)
    plot_model_df = model_df[model_df["unit_pnl"] != 0]
    plot_model_df.reset_index(drop=True, inplace=True)

    plt.plot(
        plot_model_df["unit_pnl"].cumsum() + 100,
        label=f"{model_str}_{target_str}_{betting_fnc}",
    )
    plt.xlabel("Games")
    plt.ylabel("Total Units")
    plt.title("Model Betting Strategy Performance")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_pnl_comparison(
    model_str: str,
    baseline_str: str = "model_4_2_25",
    target_str: str = "total",
    betting_fncs: Tuple[str] = ("spread_probs", "spread_probs"),
):
    """
    Plots the comparative net unit pnl given a model and a baseline.

    Args:
        model_str (str): Model prefix.
        baseline_str (str): Baseline model prefix.
        target_str (str, optional): For model, the target column. Defaults to "total".
        betting_fncs (Tuple[str], optional): Betting functions. Defaults to "spread_probs" for both.
    """
    plot_model_df = load_pkl_if_exists(
        model_str, target_str, betting_fncs[0], "odds_df"
    )
    plot_baseline_df = load_pkl_if_exists(
        baseline_str, target_str, betting_fncs[1], "odds_df"
    )

    plot_model_df.fillna(0, inplace=True)
    plot_model_df = plot_model_df[plot_model_df["unit_pnl"] != 0]
    plot_model_df.reset_index(drop=True, inplace=True)
    plot_baseline_df.fillna(0, inplace=True)
    plot_baseline_df = plot_baseline_df[plot_baseline_df["unit_pnl"] != 0]
    plot_baseline_df.reset_index(drop=True, inplace=True)

    plt.plot(
        plot_model_df["unit_pnl"].cumsum() + 100,
        label=f"{model_str}_{target_str}_{betting_fncs[0]}",
    )
    plt.plot(
        plot_baseline_df["unit_pnl"].cumsum() + 100,
        label=f"{baseline_str}_{target_str}_{betting_fncs[1]}",
    )
    plt.xlabel("Games")
    plt.ylabel("Total Units")
    plt.title("Betting Strategy Performance")
    plt.grid(True)
    plt.legend()
    plt.show()


def get_pred_metrics(
    model_str: str,
    target_str: str = "total",
    betting_fnc: str = "spread_probs",
) -> pd.DataFrame:
    """
    Creates a DataFrame with necessary betting and prediction metrics of the model.

    Args:
        model_str (str): Model prefix.
        target_str (str, optional): Target column. Defaults to "total".
        betting_fnc (str, optional): Function to determine bets. Defaults to "spread_probs".

    Returns:
        pd.DataFrame: DataFrame with different model metrics.
    """
    model_df = load_pkl_if_exists(model_str, target_str, betting_fnc, "odds_df")
    model_df.dropna(inplace=True, subset=[target_str, "pred"])
    y = model_df[target_str].values
    y_hat = model_df["pred"].values

    mae = mean_absolute_error(y, y_hat)
    mse = mean_squared_error(y, y_hat)
    r2 = r2_score(y, y_hat)

    bet_results = model_df["unit_pnl"].dropna()
    bet_results = bet_results[bet_results != 0]
    total_units = bet_results.cumsum() + 100
    net_pnl = bet_results.sum()

    sharpe = net_pnl / total_units.std()

    max_drawdown = min(bet_results.cumsum())
    percent_winning = (bet_results > 0).sum() / len(bet_results)
    num_bets = (bet_results != 0).sum()

    metrics = {
        "Mean Average Error": mae,
        "Mean Squared Error": mse,
        "R-Squared": r2,
        "Sharpe": sharpe,
        "Net PNL": net_pnl,
        "Max Drawdown": max_drawdown,
        "Number of Bets": num_bets,
        "Winning Bet %": percent_winning,
    }
    metric_df = pd.DataFrame(metrics, index=[f"{model_str}_{target_str}_{betting_fnc}"])
    metric_df.index.name = "model"
    return metric_df


def plot_model_metrics(
    model_str: str,
    baseline_str: str = None,
    target_str: str = "total",
) -> pd.DataFrame:
    """
    Plots feature importances.

    Args:
        model_str (str): Model prefix.
        baseline_str (str, optional): If exists, plots comparison to baseline.
        target_str (str, optional): Target column. Defaults to "total".

    Returns:
        pd.DataFrame: DataFrame with different model metrics.
    """
    model_pipeline = load_pkl_if_exists(model_str, target_str, file_type="pipeline")
    model = model_pipeline.named_steps["light_gbm"]
    if baseline_str:
        baseline_pipeline = load_pkl_if_exists(
            baseline_str, target_str, file_type="pipeline"
        )
        baseline = baseline_pipeline.named_steps["light_gbm"]
        _, axes = plt.subplots(1, 2, figsize=(14, 6))
        lgb.plot_importance(
            model, importance_type="gain", max_num_features=10, ax=axes[0]
        )
        axes[0].set_title(f"Feature Importance (Gain) - {model_str}")
        lgb.plot_importance(
            baseline, importance_type="gain", max_num_features=10, ax=axes[1]
        )
        axes[1].set_title(f"Feature Importance (Gain) - {baseline_str}")

        axes[0].tick_params(axis="x", labelrotation=45)
        axes[0].tick_params(axis="y", labelrotation=45)
        axes[1].tick_params(axis="x", labelrotation=45)
        axes[1].tick_params(axis="y", labelrotation=45)
        plt.tight_layout()
        plt.show()
    else:
        # If no baseline_str, just plot the feature importance for the main model
        lgb.plot_importance(model, importance_type="gain", max_num_features=10)
        plt.title(f"Feature Importance (Gain) - {model_str}")
        plt.show()


def compare_models(
    model_str: str,
    baseline_str: str = "model_4_2_25",
    target_str: str = "total",
    betting_fncs: Tuple[str] = ("spread_probs", "spread_probs"),
):
    plot_pnl_comparison(model_str, baseline_str, target_str, betting_fncs)
    metric_df = pd.concat(
        [
            get_pred_metrics(model_str, target_str, betting_fncs[0]),
            get_pred_metrics(baseline_str, target_str, betting_fncs[1]),
        ]
    )
    display(metric_df)
    plot_model_metrics(model_str, baseline_str, target_str)
