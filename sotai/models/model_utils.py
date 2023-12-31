"""Common utilities for constructing calibrated models."""
from typing import List, Optional, Union

import torch
import numpy as np

from ..enums import (
    CategoricalCalibratorInit,
    FeatureType,
    Monotonicity,
    NumericalCalibratorInit,
)
from ..features import CategoricalFeature, NumericalFeature
from ..layers.categorical_calibrator import CategoricalCalibrator
from ..layers.numerical_calibrator import NumericalCalibrator


def initialize_feature_calibrators(
    features: List[Union[NumericalFeature, CategoricalFeature]],
    output_min: Optional[float] = None,
    output_max: Union[Optional[float], List[float]] = None,
) -> torch.nn.ModuleDict:
    """Helper function to initialize calibrators for calibrated model.

    Args:
        features: A list of numerical and/or categorical feature configs.
        output_min: The minimum output value for the model. If `None`, the minimum
            output value will be unbounded.
        output_max: A list of maximum output value for each feature of the model. If
            `None`, the maximum output value will be unbounded. If a singular value, it
            will be taken as the maximum of all features.

    Returns:
        A `torch.nn.ModuleDict` of calibrators accessible by each feature's name.

    Raises:
        ValueError: If any feature configs are not `NUMERICAL` or `CATEGORICAL`.
    """
    calibrators = torch.nn.ModuleDict()
    if not isinstance(output_max, list):
        output_max = [output_max] * len(features)
    for feature, feature_max in zip(features, output_max):
        if feature.feature_type == FeatureType.NUMERICAL:
            calibrators[feature.feature_name] = NumericalCalibrator(
                input_keypoints=feature.input_keypoints,
                missing_input_value=feature.missing_input_value,
                output_min=output_min,
                output_max=feature_max,
                monotonicity=feature.monotonicity,
                kernel_init=NumericalCalibratorInit.EQUAL_SLOPES,
                projection_iterations=feature.projection_iterations,
            )
        elif feature.feature_type == FeatureType.CATEGORICAL:
            calibrators[feature.feature_name] = CategoricalCalibrator(
                num_categories=len(feature.categories),
                missing_input_value=feature.missing_input_value,
                output_min=output_min,
                output_max=feature_max,
                monotonicity_pairs=feature.monotonicity_index_pairs,
                kernel_init=CategoricalCalibratorInit.UNIFORM,
            )
        else:
            raise ValueError(
                f"Unknown feature type {feature.feature_type} for feature "
                f"{feature.feature_name}"
            )
    return calibrators


def initialize_monotonicities(
    features: List[Union[NumericalFeature, CategoricalFeature]]
) -> List[Monotonicity]:
    """Helper function to initialize monotonicities for calibrated model.

    Args:
        features: A list of numerical and/or categorical feature configs.

    Returns:
        A list of `Monotonicity.NONE` or `Monotonicity.INCREASING` based on whether
        each feature has a monotonicity or not.
    """
    monotonicities = [
        Monotonicity.NONE
        if (
            feature.feature_type == FeatureType.CATEGORICAL
            and not feature.monotonicity_pairs
        )
        or (
            feature.feature_type == FeatureType.NUMERICAL
            and feature.monotonicity == Monotonicity.NONE
        )
        else Monotonicity.INCREASING
        for feature in features
    ]
    return monotonicities


def initialize_output_calibrator(
    monotonic: bool,
    output_calibration_num_keypoints: Optional[int],
    output_min: Optional[float] = None,
    output_max: Optional[float] = None,
) -> Optional[NumericalCalibrator]:
    """Helper function to initialize output calibrator for calibrated model.

    Args:
        monotonic: Whether output calibrator should have monotonicity constraint.
        output_calibration_num_keypoints: The number of keypoints in output
            calibrator. If `0` or `None`, no output calibrator will be returned.
        output_min: The minimum output value for the model. If `None`, the minimum
            output value will be unbounded.
        output_max: The maximum output value for the model. If `None`, the maximum
            output value will be unbounded.

    Returns:
        A `torch.nn.ModuleDict` of calibrators accessible by each feature's name.

    Raises:
        ValueError: If any feature configs are not `NUMERICAL` or `CATEGORICAL`.
    """
    if output_calibration_num_keypoints:
        output_calibrator = NumericalCalibrator(
            input_keypoints=np.linspace(0.0, 1.0, num=output_calibration_num_keypoints),
            missing_input_value=None,
            output_min=output_min,
            output_max=output_max,
            monotonicity=Monotonicity.INCREASING if monotonic else Monotonicity.NONE,
            kernel_init=NumericalCalibratorInit.EQUAL_HEIGHTS,
        )
        return output_calibrator
    return None


def calibrate_and_stack(
    x: torch.Tensor, calibrators: torch.nn.ModuleDict # pylint: disable=invalid-name
) -> torch.Tensor:
    """Helper function to run calibrators along columns of given data.

    Args:
        x: The input tensor of feature values of shape `(batch_size, num_features)`.
        calibrators: A dictionary of calibrator functions.

    Returns:
        A torch.Tensor resulting from applying the calibrators and stacking the results.
    """
    return torch.column_stack(
        tuple(
            calibrator(x[:, i, None])
            for i, calibrator in enumerate(calibrators.values())
        )
    )
