import pandas as pd


class Feature:
    """
    Base class that handles different types of features to be built.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def transform(self) -> pd.DataFrame:
        """
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Each subclass must implement the transform method.")


class Preprocessing(Feature):
    """
    Handles preprocessing steps such as missing values, scaling, and encoding.

    Inherits from:
        Feature: The base feature engineering class.
    """

    def clean_cols(self):
        # Gets rid of columns with no predictive value and fixes typing
        useless_cols = [
            # "id",
            "start_time_tbd",
            "completed",
            "home_id",
            "home_pregame_win_probability",
            "home_postgame_win_probability",
            "home_postgame_elo",
            "away_id",
            "away_pregame_win_probability",
            "away_postgame_win_probability",
            "away_postgame_elo",
            "venue_id",
            "venue",
            "highlights",
            "notes",
            "id_venue",
            "name",
            "city",
            "state",
            "zip",
            "countrycode",
        ]
        self.df = self.df.drop(
            columns=[col for col in useless_cols if col in self.df.columns]
        )
        self.df["neutral_site"] = self.df["neutral_site"].apply(int)
        self.df["conference_game"] = self.df["conference_game"].apply(int)
        self.df.fillna(0)
        self.df.sort_values(by="start_date")

    def remove_nan_rows(self):
        # Gets rid of empty home and away points and quarterly data
        self.df = self.df.dropna(
            subset=[
                "home_points",
                "away_points",
                "home_line_scores",
                "away_line_scores",
                "home_days_since_last_game",
                "away_days_since_last_game",
            ]
        )
        self.df = self.df[
            self.df["home_line_scores"].apply(lambda x: len(x) >= 4)
            & self.df["away_line_scores"].apply(lambda x: len(x) >= 4)
        ]
        # TODO: for each nan attendance, for every team, average the attendance for filling in NaN's all time
        # TODO: for each nan elo & excitement index, for every team, get a rolling average of the previous.
        # for now, just to get things going, we fill with 0
        self.df.fillna(0, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def encode_categorical_cols(self):
        # One hot encodes, then drops all categorical columns
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
        # Splits out the list of scores into four columns
        self.df["ot"] = self.df["home_line_scores"].apply(lambda x: int(len(x) > 4))
        for prefix in ["home", "away"]:
            line_score = f"{prefix}_line_scores"
            self.df[line_score] = self.df[line_score].apply(
                lambda x: x[0:4] + [sum(x[5:])] if len(x) >= 5 else x + [0]
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
            self.df.drop(
                [
                    line_score,
                    f"{prefix}_points",
                    f"{prefix}_q1",
                    f"{prefix}_q2",
                    f"{prefix}_q3",
                    f"{prefix}_q4",
                ],
                axis=1,
                inplace=True,
            )

    def date_to_days_since(self):
        # Create df_home and df_away as before
        df_home = self.df[["start_date", "home_team"]].rename(
            columns={"home_team": "team"}
        )
        df_away = self.df[["start_date", "away_team"]].rename(
            columns={"away_team": "team"}
        )

        # Concatenate home and away data to create a single list of all games for each team
        df_combined = pd.concat([df_home, df_away])

        # Sort by team and start_date
        df_combined = df_combined.sort_values(by=["team", "start_date"])

        # Calculate the previous game for each team
        df_combined["previous_game"] = df_combined.groupby("team")["start_date"].shift(
            1
        )

        # Merge the previous game data for home team
        self.df["previous_game"] = None
        for side in ["home", "away"]:
            self.df = self.df.merge(
                df_combined[["team", "start_date", "previous_game"]],
                left_on=[f"{side}_team", "start_date"],
                right_on=["team", "start_date"],
                how="left",
                suffixes=("", f"_{side}"),
            )
            self.df[f"{side}_days_since_last_game"] = (
                self.df["start_date"] - self.df[f"previous_game_{side}"]
            ).apply(lambda x: x.days if pd.notna(x) else None)

        # Drop temporary columns from merge
        self.df.drop(
            columns=[
                "team",
                "previous_game",
                "previous_game_home",
                "team_away",
                "previous_game_away",
                # "start_date",
            ],
            inplace=True,
        )

    def pred_ou(self):
        self.df["total"] = self.df["home_points"] + self.df["away_points"]
        self.df.drop(
            columns=[
                "home_points",
                "away_points",
                "home_line_scores",
                "away_line_scores",
            ],
            inplace=True,
        )

    def transform(self) -> pd.DataFrame:
        self.clean_cols()
        self.date_to_days_since()  # Goes before encoding
        self.remove_nan_rows()  # Goes after date to days since, removes first instance of a team (no data)
        self.encode_categorical_cols()
        # self.expand_quarters()
        self.pred_ou()
        return self.df


class OffensiveFeatures(Feature):
    # TODO: Kalman filter for scoring in each quarter
    # TODO: Percent of yards come from rushing/passing - both Offense and Defense
    pass


class DefensiveFeatures(Feature):
    pass


class NeutralSiteFeatures(Feature):
    # TODO: is bowl game?
    pass


class FeaturePipeline:
    """
    Orchestrates the feature engineering process in order.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Provides the steps for the pipeline.

        Args:
            df (pd.DataFrame): Data and venue merged DataFrame.
        """
        self.df = df
        self.steps = [
            Preprocessing(df),
            # OffensiveFeatures(df),
            # DefensiveFeatures(df),
            # NeutralSiteFeatures(df),
        ]

    def run(self) -> pd.DataFrame:
        """
        Executes all feature engineering steps.
        """
        for step in self.steps:
            self.df = step.transform()
        return self.df
