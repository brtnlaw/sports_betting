import pandas as pd
import numpy as np
from model.feature_engineering import generate_features
from sklearn.neighbors import KernelDensity


def build_kde(
    data: pd.DataFrame,
    input_data: pd.DataFrame,
    features: list,
    weights: list,
    kernel: str = "epanechnikov",
    bandwidth: int = 1,
) -> KernelDensity:
    """
    Builds KDE using input data and data.

    Args:
        data (pd.DataFrame): Training data.
        input_data (pd.DataFrame): New data row which we change features.
        features (list): List of features.
        weights (list): Weights corresponding to features.
        kernel (str, optional): Kernel for KDE. Defaults to "epanechnikov".
        bandwidth (int, optional): Bandwidth parameter. Defaults to 1.

    Returns:
        KernelDensity: KDE object.
    """
    data_with_features = generate_features(data, input_data)
    sample_weight = (data[features] * weights).sum(axis=1)
    pra_df = data_with_features[["points", "total_rebounds", "assists"]]
    pra_array = [np.array(row) for row in pra_df.values]
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
        pra_array, sample_weight=sample_weight
    )
    return kde
