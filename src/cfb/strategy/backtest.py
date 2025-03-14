import pandas as pd
import matplotlib.pyplot as plt
import warnings
from data_preprocessing import Preprocessor
import strategy.betting_logic as betting_logic
from train import train_model, evaluate_model_metrics

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


def cross_validate(df, X, y):
    """Performs rolling cross-validation using past seasons to predict future ones."""
    cv_year_indices = df.groupby("season").head(1).index
    init_train_yrs = 5  # Number of initial seasons to train on
    cv_year_indices = cv_year_indices[init_train_yrs:].append(pd.Index([df.index[-1]]))

    for i in range(len(cv_year_indices) - 1):
        train_idx, test_idx = cv_year_indices[i], cv_year_indices[i + 1]

        # Train-test split
        X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
        X_test = X.iloc[train_idx:test_idx]

        # Train model & predict
        model = train_model(X_train, y_train)
        df.loc[train_idx : test_idx - 1, "pred"] = model.predict(X_test)

    # Apply betting logic & plot results
    df = betting_logic.simple_percentage(df)
    return model, df


def evaluate_cv(model, df):
    plot_pnl(df)
    y_pred = df["pred"]
    y_test = df["total"]

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
    # Run preprocessing pipeline
    preprocessor = Preprocessor("cfb")
    df, X, y = preprocessor.prepare_data(target_col="total")

    # Perform backtesting
    df, model = cross_validate(df, X, y)

    """
    
    print("Step 1: Loading data...")
    data_prep = DataPrep(dataset="cfb")
    raw_data = data_prep.load_data()

    # 2. Feature Engineering
    print("Step 2: Feature engineering...")
    feature_engineering = FeatureEngineering(raw_data)
    engineered_data = feature_engineering.apply_feature_engineering()

    # 3. Preprocess Data for ML (cleaning, encoding, scaling, splitting) ^ switch this
    print("Step 3: Data processing...")
    data_processing = DataProcessing(engineered_data)
    X_train, X_test, y_train, y_test = data_processing.prepare_data(target_col="total")

    # 4. Train the Model
    print("Step 4: Training the model...")
    model = Model()
    model.train(X_train, y_train)

    # 5. Evaluate the Model
    print("Step 5: Evaluating the model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    # 6. (Optional) Save the Model for Future Predictions
    model.save_model()

    """
