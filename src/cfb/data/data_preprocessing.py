import datetime as dt

import pandas as pd


class Preprocessor:
    """
    Handles preprocessing steps such as missing values, scaling, encoding, and feature transformations.
    """

    def __init__(self, raw_data, target_col):
        self.df = raw_data
        self.target_col = target_col
        self.odds_df = None
        self.X = None
        self.y = None

    def encode_categorical_cols(self):
        """One-hot encodes categorical columns and standardizes column names. Betting columns are already excluded."""
        exclude_cols = [
            "home_line_scores",
            "away_line_scores",
            "start_date",
            "highlights",
            "is_grass",
        ]
        categorical_cols = [
            col
            for col in self.df.select_dtypes(include=["object", "category"]).columns
            if col not in exclude_cols
        ]
        self.df = pd.get_dummies(self.df, columns=categorical_cols, dtype=int)
        self.df.columns = self.df.columns.str.lower().str.replace(" ", "_")

    def expand_quarters(self):
        """Expands quarterly game scores into separate columns."""
        self.df = self.df[
            self.df["home_line_scores"].apply(lambda x: len(x) >= 4)
            & self.df["away_line_scores"].apply(lambda x: len(x) >= 4)
        ]
        self.df["ot"] = self.df["home_line_scores"].apply(lambda x: int(len(x) > 4))

        for prefix in ["home", "away"]:
            line_score = f"{prefix}_line_scores"
            self.df[line_score] = self.df[line_score].apply(
                lambda x: x[:4] + [sum(x[5:])] if len(x) >= 5 else x + [0]
            )
            self.df[
                [
                    f"{prefix}_q1",
                    f"{prefix}_q2",
                    f"{prefix}_q3",
                    f"{prefix}_q4",
                    f"{prefix}_ot",
                ]
            ] = pd.DataFrame(self.df[line_score].tolist(), index=self.df.index)

            self.df[f"{prefix}_h1"] = self.df[f"{prefix}_q1"] + self.df[f"{prefix}_q2"]
            self.df[f"{prefix}_h2"] = self.df[f"{prefix}_q3"] + self.df[f"{prefix}_q4"]

            self.df.drop([line_score, f"{prefix}_points"], axis=1, inplace=True)

    def date_to_days_since(self):
        """Calculates days since last game for each team."""
        df_home = self.df[["start_date", "home_team"]].rename(
            columns={"home_team": "team"}
        )
        df_away = self.df[["start_date", "away_team"]].rename(
            columns={"away_team": "team"}
        )
        df_combined = pd.concat([df_home, df_away]).sort_values(
            by=["team", "start_date"]
        )
        df_combined["previous_game"] = df_combined.groupby("team")["start_date"].shift(
            1
        )

        self.df["previous_game"] = None
        for side in ["home", "away"]:
            self.df = self.df.merge(
                df_combined[["team", "start_date", "previous_game"]],
                left_on=[f"{side}_team", "start_date"],
                right_on=["team", "start_date"],
                how="left",
                suffixes=("", f"_{side}"),
            )
            self.df[f"previous_game_{side}"].fillna(dt.date(2000, 1, 1), inplace=True)
            self.df[f"{side}_days_since_last_game"] = (
                self.df["start_date"] - self.df[f"previous_game_{side}"]
            ).apply(lambda x: x.days)
        self.df.drop(
            columns=[
                "team",
                "previous_game",
                "previous_game_home",
                "previous_game_away",
            ],
            inplace=True,
        )

    def add_total_score(self):
        """Calculates total score for Over/Under betting."""
        self.df["total"] = self.df["home_points"] + self.df["away_points"]
        self.df.drop(
            columns=[
                "home_points",
                "away_points",
            ],
            inplace=True,
        )

    # ===============================
    # Final steps
    # ===============================

    def remove_nan_rows(self):
        """Removes rows with missing game data."""
        self.df["neutral_site"] = self.df["neutral_site"].astype(int)
        self.df["conference_game"] = self.df["conference_game"].astype(int)
        self.df.dropna(
            subset=[
                "home_days_since_last_game",
                "away_days_since_last_game",
            ],
            inplace=True,
        )
        self.df.sort_values(by="start_date", inplace=True)
        self.df.fillna(0, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def split_odds_X_y(self):
        """Keeping the index together, splits into the odds, X, and y"""
        feature_cols = [
            col
            for col in self.df.columns
            if col not in ["id", "season", "start_date", self.target_col]
        ]
        # The only time we should be setting the index
        self.df.set_index("id", inplace=True)
        # Note that the odds_df is only directly what we're trying to predict. Adjacent odds data is permissible.
        betting_cols = ["min_ou", "max_ou"]
        self.odds_df = self.df[[self.target_col] + betting_cols]
        self.X = self.df[feature_cols]
        self.y = self.df[self.target_col]

    def preprocess_data(self):
        """Runs the full preprocessing pipeline and returns cleaned feature-target split."""
        self.date_to_days_since()
        self.encode_categorical_cols()
        self.add_total_score()
        self.remove_nan_rows()
        self.split_odds_X_y()
        return self.odds_df, self.X, self.y
