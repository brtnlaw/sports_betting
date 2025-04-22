# Util files with different ways to bet for ease of testing


def _apply_edge(
    df, condition, success_condition, failure_condition, units, payout=0.87
):
    # TODO: replace with actual lines.
    df.loc[condition & success_condition, "unit_pnl"] = payout * units
    df.loc[condition & failure_condition, "unit_pnl"] = -1 * units
    return df


def simple_percentage(df):
    # Initialize column in case none of the conditions match
    df["unit_pnl"] = 0

    over = (df["max_ou"] != 0) & (df["pred"] > 1.1 * df["max_ou"])
    df = _apply_edge(
        df, over, (df["total"] > df["max_ou"]), (df["total"] < df["max_ou"]), 1
    )

    under = (df["min_ou"] != 0) & (df["pred"] < 0.9 * df["min_ou"])
    df = _apply_edge(
        df, under, (df["total"] < df["min_ou"]), (df["total"] > df["min_ou"]), 1
    )
    return df


def simple_stellage(df):
    # The greater the edge, the larger the size up
    df["unit_pnl"] = 0

    over1 = (df["max_ou"] != 0) & (df["pred"] > 1.1 * df["max_ou"])
    over2 = (df["max_ou"] != 0) & (df["pred"] > 1.2 * df["max_ou"])
    df = _apply_edge(
        df, over1, (df["total"] > df["max_ou"]), (df["total"] < df["max_ou"]), 1
    )
    df = _apply_edge(
        df, over2, (df["total"] > df["max_ou"]), (df["total"] < df["max_ou"]), 2
    )

    under1 = (df["min_ou"] != 0) & (df["pred"] < 0.9 * df["min_ou"])
    under2 = (df["min_ou"] != 0) & (df["pred"] < 0.8 * df["min_ou"])
    df = _apply_edge(
        df, under1, (df["total"] < df["min_ou"]), (df["total"] > df["min_ou"]), 1
    )
    df = _apply_edge(
        df, under2, (df["total"] < df["min_ou"]), (df["total"] > df["min_ou"]), 2
    )
    return df


def kelly_criterion(df):
    # kelly_size = p - (1-q)/b
    pass


# Brier Score
