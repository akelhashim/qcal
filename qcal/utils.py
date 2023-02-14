"""
Helper functions.
"""

import pandas as pd


def load(path: str) -> pd.DataFrame:
    """Load a saved CircuitSet

    Args:
        path (str): filepath for saved CircuitSet.

    Returns:
        pd.DataFrame: CircuitSet
    """
    return pd.read_pickle(path)