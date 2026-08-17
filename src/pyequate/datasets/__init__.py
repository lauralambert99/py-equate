# -*- coding: utf-8 -*-
"""
Bundled example datasets for pyequate.

Data files live alongside this module as CSVs and are loaded via
``importlib.resources`` so they work whether the package is imported
from source or from an installed wheel.
"""

from importlib.resources import files

import pandas as pd

__all__ = ["load_dataset", "list_datasets"]


def list_datasets():
    """
    List the names of the datasets bundled with pyequate.

    Returns:
    list of str
        Dataset names (without the ``.csv`` extension), sorted.
    """
    return sorted(
        p.name[:-4]
        for p in files("pyequate.datasets").iterdir()
        if p.name.endswith(".csv")
    )


def load_dataset(name):
    """
    Load a bundled dataset by name.

    Parameters:
    name : str
        Dataset name without the ``.csv`` extension, e.g. ``"ACTmath"``.
        Use :func:`list_datasets` to see the available names.

    Returns:
    DataFrame
        The dataset as a pandas DataFrame.

    Raises:
    FileNotFoundError
        If no bundled dataset matches ``name``.
    """
    resource = files("pyequate.datasets") / f"{name}.csv"
    if not resource.is_file():
        available = ", ".join(list_datasets())
        raise FileNotFoundError(
            f"No bundled dataset named {name!r}. Available datasets: {available}"
        )
    with resource.open("r") as f:
        return pd.read_csv(f)
