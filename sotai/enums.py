"""Enum Classes for SOTAI SDK."""
from enum import Enum, EnumMeta
from typing import Any


class _Metaclass(EnumMeta):
    """Base `EnumMeta` subclass for accessing enum members directly."""

    def __getattribute__(cls, __name: str) -> Any:
        value = super().__getattribute__(__name)
        if isinstance(value, Enum):
            value = value.value
        return value


class _Enum(str, Enum, metaclass=_Metaclass):
    """Base Enum Class."""


class TargetType(_Enum):
    """The type of target to predict.

    - CLASSIFICATION: classification target i.e. binary 0/1.
    - REGRESSION: regression target i.e. continuous float.
    """

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class LossType(_Enum):
    """The type of loss function to use.

    - BINARY_CROSSENTROPY: binary cross entropy loss.
    - HINGE: hinge loss.
    - HUBER: huber loss.
    - MAE: mean absolute error loss.
    - MSE: mean squared error loss.
    """

    BINARY_CROSSENTROPY = "binary"
    HINGE = "hinge"
    HUBER = "huber"
    MAE = "mae"
    MSE = "mse"


class Metric(_Enum):
    """The type of metric to use.

    - AUC: area under the ROC curve.
    - MAE: mean absolute error.
    - MSE: mean squared error.
    """

    AUC = "auc"
    MAE = "mean_absolute_error"
    MSE = "mean_squared_error"


class InputKeypointsInit(_Enum):
    """Type of initialization to use for NumericalCalibrator input keypoints.

    - QUANTILES: initialize the input keypoints such that each segment will see the same
        number of examples.
    - UNIFORM: initialize the input keypoints uniformly spaced in the feature range.
    """

    QUANTILES = "quantiles"
    UNIFORM = "uniform"


class InputKeypointsType(_Enum):
    """The type of input keypoints to use.

    - FIXED: the input keypoints will be fixed during initialization.
    """

    FIXED = "fixed"
    # TODO: add learned interior functionality
    # LEARNED = "learned_interior"


class FeatureType(_Enum):
    """Type of feature.

    - UNKNOWN: a feature with a type that our system does not currently support.
    - NUMERICAL: a numerical feature that should be calibrated using an instance of
        `NumericalCalibrator`.
    - CATEGORICAL: a categorical feature that should be calibrated using an instance of
        `CategoricalCalibrator`.
    """

    UNKNOWN = "unknown"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


class NumericalCalibratorInit(_Enum):
    """Type of kernel initialization to use for NumericalCalibrator.

    - EQUAL_HEIGHTS: initialize the kernel such that all segments have the same height.
    - EQUAL_SLOPES: initialize the kernel such that all segments have the same slope.
    """

    EQUAL_HEIGHTS = "equal_heights"
    EQUAL_SLOPES = "equal_slopes"


class CategoricalCalibratorInit(_Enum):
    """Type of kernel initialization to use for CategoricalCalibrator.

    - UNIFORM: initialize the kernel with uniformly distributed values. The sample range
        will be [`output_min`, `output_max`] if both are provided.
    - CONSTANT: initialize the kernel with a constant value for all categories. This
        value will be `(output_min + output_max) / 2` if both are provided.
    """

    UNIFORM = "uniform"
    CONSTANT = "constant"


class Monotonicity(_Enum):
    """Type of monotonicity constraint.

    - NONE: no monotonicity constraint.
    - INCREASING: increasing monotonicity i.e. increasing input increases output.
    - DECREASING: decreasing monotonicity i.e. increasing input decreases output.
    """

    NONE = "none"
    INCREASING = "increasing"
    DECREASING = "decreasing"


class APIStatus(_Enum):
    """Status of API.

    - SUCCESS: API call was successful
    - ERROR: API call was unsuccessful
    """

    SUCCESS = "success"
    ERROR = "error"


class InferenceConfigStatus(_Enum):
    """Enum for InferenceConfig status.

    - FAILED: inference job failed.
    - INITIALIZING: inference job is initializing.
    - PREPARING: inference job is preparing to run.
    - RUNNING: inference job is running.
    - SUCCESS: inference job completed successfully.
    """

    FAILED = "failed"
    INITIALIZING = "initializing"
    PREPARING = "preparing"
    RUNNING = "running"
    SUCCESS = "success"
