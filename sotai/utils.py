"""Utility functions for pipelines."""
import logging
import math
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .enums import FeatureType
from .types import CategoricalFeatureConfig, NumericalFeatureConfig


def _determine_feature_types(
    data: pd.DataFrame,
    target: str,
    categories: Optional[List[str]] = None,
) -> Dict[str, FeatureType]:
    """Determines the type of feature data.

    Args:
        data: The data to be used for training.
        target: The name of the target column.
        categories: The column names in `data` for categorical columns.

    Returns:
        A dictionary mapping column names to their corresponding feature type.
    """
    feature_types = {}
    for column in data.columns:
        if column == "target":
            continue
        dtype_kind = data[column].dtype.kind
        if dtype_kind in ["S", "O", "b"]:  # string, object, boolean
            feature_types[column] = FeatureType.CATEGORICAL
        elif dtype_kind in ["i", "u", "f"]:  # integer, unsigned integer, float
            feature_types[column] = FeatureType.NUMERICAL
        else:  # datetime, timedelta, complex, etc.
            feature_types[column] = FeatureType.UNKNOWN
    return feature_types


def default_feature_configs(
    data: pd.DataFrame,
    target: str,
    categories: Optional[List[str]] = None,
) -> Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]]:
    """Generates default feature configs for the given data and target.

    Args:
        data: The data to be used for training.
        target: The name of the target column.
        feature_types: A dictionary mapping column names to their corresponding feature
            type.

    Returns:
        A dictionary mapping column names to their corresponding feature config.
    """
    feature_configs = {}
    feature_types = _determine_feature_types(data, target, categories)
    for feature_name, feature_type in feature_types.items():
        if feature_type == FeatureType.CATEGORICAL:
            feature_configs[feature_name] = CategoricalFeatureConfig(
                name=feature_name,
                categories=data[feature_name].dropna().unique().tolist(),
            )
        elif feature_type == FeatureType.NUMERICAL:
            feature_configs[feature_name] = NumericalFeatureConfig(
                name=feature_name,
                missing_input_value=math.floor(data[feature_name].min() - 1),
            )
        else:
            logging.info("Skipping feature %s with unknown type.", feature_name)
            continue
    return feature_configs
