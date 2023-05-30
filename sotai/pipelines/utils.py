"""Utility functions for pipelines."""
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..enums import FeatureType, TargetType
from .types import CategoricalFeatureConfig, NumericalFeatureConfig


def determine_target_type(data: np.ndarray) -> TargetType:
    """Returns the type of a target determined from its data."""
    raise NotImplementedError()


def determine_feature_types(
    data: np.ndarray,
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
    raise NotImplementedError()


def generate_default_feature_configs(
    data: pd.DataFrame,
    target: str,
    feature_types: Dict[str, FeatureType],
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
    raise NotImplementedError()
