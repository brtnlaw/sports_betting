# Util files with different ways to bet for ease of testing
def simple_percentage(df):
    # TODO: replace once I get an actual line
    # Assumes a -115 on each side == 1.87
    # Apply betting logic using .loc to avoid SettingWithCopyWarning
    df.loc[
        (df["pred"] > 1.1 * df["max_ou"]) & (df["total"] > df["max_ou"]), "unit_pnl"
    ] = 0.87
    df.loc[
        (df["pred"] > 1.1 * df["max_ou"]) & (df["total"] < df["max_ou"]), "unit_pnl"
    ] = -1
    df.loc[
        (df["pred"] < 0.9 * df["min_ou"]) & (df["total"] < df["min_ou"]), "unit_pnl"
    ] = 0.87
    df.loc[
        (df["pred"] < 0.9 * df["min_ou"]) & (df["total"] > df["min_ou"]), "unit_pnl"
    ] = -1
    return df
