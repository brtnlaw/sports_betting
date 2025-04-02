# Util files with different ways to bet for ease of testing
def simple_percentage(df):
    # TODO: replace once I get an actual line
    # Assumes a -115 on each side == 1.87
    conditions = [
        (
            (df["max_ou"] != 0)
            & (df["pred"] > 1.1 * df["max_ou"])
            & (df["total"] > df["max_ou"]),
            0.87,
        ),
        (
            (df["max_ou"] != 0)
            & (df["pred"] > 1.1 * df["max_ou"])
            & (df["total"] < df["max_ou"]),
            -1,
        ),
        (
            (df["min_ou"] != 0)
            & (df["pred"] < 0.9 * df["min_ou"])
            & (df["total"] < df["min_ou"]),
            0.87,
        ),
        (
            (df["min_ou"] != 0)
            & (df["pred"] < 0.9 * df["min_ou"])
            & (df["total"] > df["min_ou"]),
            -1,
        ),
    ]

    for condition, value in conditions:
        df.loc[condition, "unit_pnl"] = value

    return df


def kelly_criterion(df):
    # kelly_size = p - (1-q)/b
    pass


# Brier Score
