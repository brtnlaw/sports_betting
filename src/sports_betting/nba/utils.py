import hashlib
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import psycopg2
import warnings
import yaml
from sklearn.neighbors import KernelDensity
from typing import List, Optional

TEAM_CODE_DICT = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def group_contiguous_dates(dates: List[str]) -> List[str]:
    """
    Groups list of dates into ranges.

    Args:
        dates (List[str]): List of dates to group.

    Returns:
        List[str]: List of ranges.
    """
    # Sort the dates
    sorted_dates = sorted(set(dates))

    # Group contiguous dates including weekends and ignoring two-day gaps
    groups = []
    group = [sorted_dates[0]]
    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] - group[-1] <= dt.timedelta(days=2):
            group.append(sorted_dates[i])
        else:
            groups.append(group)
            group = [sorted_dates[i]]
    groups.append(group)

    # Format the groups
    formatted_groups = []
    for group in groups:
        if len(group) == 1:
            formatted_groups.append(group[0].strftime("%m/%d/%y"))
        else:
            formatted_groups.append(
                f"{group[0].strftime('%m/%d/%y')}-{group[-1].strftime('%m/%d/%y')}"
            )
    return formatted_groups


def visualize_kde(kde: KernelDensity, data: pd.DataFrame) -> None:
    x = np.arange(data["points"].min(), data["points"].max() + 1)
    y = np.arange(data["total_rebounds"].min(), data["total_rebounds"].max() + 1)
    z = np.arange(data["assists"].min(), data["assists"].max() + 1)
    X, Y, Z = np.meshgrid(x, y, z)

    # Reshape the meshgrid to a list of coordinates
    coordinates = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    pmf_values = np.exp(kde.score_samples(coordinates))
    # reconstruct pmf
    pmf = np.zeros((len(x), len(y), len(z)))
    for idx, val in zip(coordinates, pmf_values):
        pmf[idx[0], idx[1], idx[2]] = val

    _, ax = plt.subplots(3, 1, figsize=(18, 10))

    ax[0].plot(x, np.sum(pmf, axis=(1, 2)))
    ax[0].set_title("points pmf")
    ax[0].set_xlabel("points")
    ax[0].set_ylabel("density")

    ax[1].plot(y, np.sum(pmf, axis=(0, 2)))
    ax[1].set_title("rebounds pmf")
    ax[1].set_xlabel("rebounds")
    ax[1].set_ylabel("density")

    ax[2].plot(z, np.sum(pmf, axis=(0, 1)))
    ax[2].set_title("assists pmf")
    ax[2].set_xlabel("assists")
    ax[2].set_ylabel("density")

    plt.tight_layout()
    plt.show()
