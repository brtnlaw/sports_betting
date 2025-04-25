import numpy as np
import pandas as pd
from scipy.stats import norm

from db_utils import retrieve_data


class BettingLogic:
    SPREAD_SD = 22
    COND_SPREAD_SD = 15

    def __init__(self, betting_fnc: str = "spread_cond_betting"):
        self.betting_fnc = betting_fnc
        self.spread_cover_probs = self._initialize_cond_probs()
        self.spread_sd = self.SPREAD_SD
        self.cond_spread_sd = self.COND_SPREAD_SD

    def _apply_edge(
        self,
        odds_df,
        condition,
        success_condition,
        failure_condition,
        units,
        payout=0.87,
    ):
        # TODO: replace with actual lines.
        odds_df.loc[condition & success_condition, "unit_pnl"] = payout * units
        odds_df.loc[condition & failure_condition, "unit_pnl"] = -1 * units
        return odds_df

    def _initialize_cond_probs(self) -> pd.DataFrame:
        """
        Gets conditional probability matrix. Utilizes Gaussian normal modified for higher probabilities of increments of 3 and 7.
        Essentially, this allows us to see the difference in probability between two spread projections.
        See https://arxiv.org/pdf/2212.08116.

        Returns:
            pd.DataFrame: Conditional probability matrix. Rows represent outcomes while columns represent betting lines and predictions.
        """
        hist_df = retrieve_data("cfb", "games")
        hist_df["home_away_spread"] = hist_df["away_points"] - hist_df["home_points"]
        count_col = hist_df[hist_df["home_away_spread"].abs() <= 60]["home_away_spread"]
        hist_pcts = count_col.value_counts().sort_index() / len(count_col)

        # Hard-coded, better results with a slightly larger sd
        historical_spread_range = np.arange(-60, 61)
        historical_spread_lines_range = np.arange(-40, 41)

        # Attain difference in football scores w/ Gaussian normal
        gauss_probs = norm.cdf(
            historical_spread_range + 0.5, loc=0, scale=self.spread_sd
        ) - norm.cdf(historical_spread_range - 0.5, loc=0, scale=self.spread_sd)
        gauss_probs = pd.Series(gauss_probs, index=historical_spread_range, name="norm")

        mult_df = pd.concat([hist_pcts, gauss_probs], axis=1)
        mult_df["mult"] = mult_df["count"] / mult_df["norm"]

        # Every column is the PMF of N(mu, 15), where mu is the spreads line
        cdf_dict = {
            line: norm.cdf(
                historical_spread_range + 0.5, loc=line, scale=self.cond_spread_sd
            )
            - norm.cdf(
                historical_spread_range - 0.5, loc=line, scale=self.cond_spread_sd
            )
            for line in historical_spread_lines_range
        }
        cond_df = pd.DataFrame(cdf_dict, index=historical_spread_range)

        # Multiply by factor and normalize
        cond_df = cond_df.mul(mult_df["mult"], axis=0)
        for col in cond_df:
            cond_df[col] = cond_df[col] / (cond_df[col].sum())
        return cond_df

    def _get_weighted_cover_prob(self, our_line: int, book_line: int):
        if (
            pd.isna(our_line)
            or our_line is None
            or pd.isna(book_line)
            or book_line is None
        ):
            return None

        clipped = np.clip(our_line, -40, 40)

        our_upper = int(np.ceil(clipped))
        our_lower = int(np.floor(clipped))
        our_upper_wt = our_line - our_lower
        our_lower_wt = our_upper - our_line

        book_upper = int(np.ceil(book_line))
        book_lower = int(np.floor(book_line))
        book_upper_wt = book_line - book_lower
        book_lower_wt = book_upper - book_line

        prob_cover = our_upper_wt * (
            book_upper_wt
            * self.spread_cover_probs.loc[
                self.spread_cover_probs.index <= book_upper, our_upper
            ].sum()
            + book_lower_wt
            * self.spread_cover_probs.loc[
                self.spread_cover_probs.index <= book_lower, our_upper
            ].sum()
        ) + our_lower_wt * (
            book_upper_wt
            * self.spread_cover_probs.loc[
                self.spread_cover_probs.index <= book_upper, our_lower
            ].sum()
            + book_lower_wt
            * self.spread_cover_probs.loc[
                self.spread_cover_probs.index <= book_upper, our_lower
            ].sum()
        )
        return prob_cover

    def spread_probs(self, odds_df):
        # pred of covering min conditional on our theo
        odds_df["cover_min_prob"] = odds_df.apply(
            lambda row: self._get_weighted_cover_prob(row["pred"], row["min_spread"]),
            axis=1,
        )
        odds_df["cover_max_prob"] = odds_df.apply(
            lambda row: self._get_weighted_cover_prob(row["pred"], row["max_spread"]),
            axis=1,
        )

        home_better = (odds_df["pred"] < odds_df["min_spread"]) & (
            odds_df["cover_min_prob"] > 0.60
        )
        odds_df = self._apply_edge(
            odds_df,
            home_better,
            (odds_df["home_away_spread"] < odds_df["min_spread"]),
            odds_df["home_away_spread"] > odds_df["min_spread"],
            1,
        )

        home_worse = (odds_df["pred"] > odds_df["max_spread"]) & (
            odds_df["cover_max_prob"] < 0.40
        )
        odds_df = self._apply_edge(
            odds_df,
            home_worse,
            (odds_df["home_away_spread"] > odds_df["max_spread"]),
            odds_df["home_away_spread"] < odds_df["max_spread"],
            1,
        )
        return odds_df

    def apply_bets(self, odds_df):
        return getattr(self, self.betting_fnc)(odds_df)
