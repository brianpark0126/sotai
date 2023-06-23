"""Utility functions for pipelines."""
import logging
from typing import Dict, Union

import pandas as pd

from .constants import MISSING_CATEGORY_VALUE, MISSING_NUMERICAL_VALUE
from .enums import FeatureType
from .types import CategoricalFeature, NumericalFeature


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


def replace_missing_values(
    data: pd.DataFrame, features: Dict[str, Union[CategoricalFeature, NumericalFeature]]
) -> pd.DataFrame:
    """Replaces empty values or unspecified categories with a constant value."""
    for feature_name, feature_config in features.items():
        if feature_config.type == FeatureType.CATEGORICAL:
            unspecified_categories = list(
                set(data[feature_name].unique().tolist())
                - set(feature_config.categories)
            )
            if unspecified_categories:
                logging.info(
                    "Replacing %s with %s for feature %s",
                    unspecified_categories,
                    MISSING_CATEGORY_VALUE,
                    feature_name,
                )
                data[feature_name].replace(
                    unspecified_categories, MISSING_CATEGORY_VALUE, inplace=True
                )
            data[feature_name].fillna(MISSING_CATEGORY_VALUE, inplace=True)
        else:
            data[feature_name].fillna(MISSING_NUMERICAL_VALUE, inplace=True)

    return data
