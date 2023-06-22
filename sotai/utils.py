"""Utility functions for pipelines."""
from typing import Dict

import pandas as pd

from .enums import FeatureType


def determine_feature_types(data: pd.DataFrame) -> Dict[str, FeatureType]:
    """Returns a dictionary mapping feature name to type for the given data."""
    feature_types = {}
    for column in data.columns:
        dtype_kind = data[column].dtype.kind
        if dtype_kind in ["S", "O", "b"]:  # string, object, boolean
            feature_types[column] = FeatureType.CATEGORICAL
        elif dtype_kind in ["i", "u", "f"]:  # integer, unsigned integer, float
            feature_types[column] = FeatureType.NUMERICAL
        else:  # datetime, timedelta, complex, etc.
            feature_types[column] = FeatureType.UNKNOWN
    return feature_types


def extract_linear_coefficients():
    """Extracts the linear coefficients from a calibrated linear model."""
    raise NotImplementedError()
