"""Enum Classes for SOTAI SDK"""
from enum import Enum, EnumMeta
from typing import Any


# TODO (will): Fill out docstrings for those missing details.
class _Metaclass(EnumMeta):
    """Base `EnumMeta` subclass for accessing enum members directly."""

    def __getattribute__(cls, __name: str) -> Any:
        value = super().__getattribute__(__name)
        if isinstance(value, Enum):
            value = value.value
        return value


class _Enum(str, Enum, metaclass=_Metaclass):
    """Base Enum Class"""


class TargetType(_Enum):
    """The type of target to predict."""

    UNKNOWN = "unknown"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class LossType(_Enum):
    """The type of loss function to use."""

    BINARY_CROSSENTROPY = "binary"
    HINGE = "hinge"
    HUBER = "huber"
    MEAN_ABSOLUTE_ERROR = "mae"
    MEAN_SQUARED_ERROR = "mse"


class Metric(_Enum):
    """The type of metric to use."""

    AUC = "auc"
    BINARY_ACCURACY = "binary_accuracy"
    F1 = "f1"
    MAE = "mean_absolute_error"
    MSE = "mean_squared_error"
    PRECISION = "precision"
    RECALL = "recall"
    RMSE = "root_mean_squared_error"


class ModelFramework(_Enum):
    """The type of model framework to use."""

    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"


class ModelType(_Enum):
    """The type of model to use."""

    LINEAR = "linear"
    LATTICE = "lattice"
    ENSEMBLE = "ensemble"


class CalibratorRegularizationType(_Enum):
    """The type of regularization to use for the calibrator."""

    LAPLACIAN = "laplacian"
    HESSIAN = "hessian"
    WRINKLE = "wrinkle"


class LatticeRegularizationType(_Enum):
    """The type of regularization to use for the lattice."""

    LAPLACIAN = "laplacian"
    TORSION = "torsion"


class Interpolation(_Enum):
    """The type of interpolation to use for the lattice."""

    HYPERCUBE = "hypercube"
    SIMPLEX = "simplex"


class Parameterization(_Enum):
    """The type of parameterization to use for the lattice."""

    ALL_VERTICES = "all_vertices"
    KFL = "kronecker_factored"


class EnsembleType(_Enum):
    """The type of ensemble to use."""

    RANDOM = "random"
    RTL = "rtl_layer"
    CRYSTALS = "crystals"


class InputKeypointsInit(_Enum):
    """Type of initialization to use for NumericalCalibrator input keypoints.

    - QUANTILES: initialize the input keypoints such that each segment will see the same
        number of examples.
    - UNIFORM: initialize the input keypoints uniformly spaced in the feature range.
    """

    QUANTILES = "quantiles"
    UNIFORM = "uniform"


class InputKeypointsType(_Enum):
    """The type of input keypoints to use."""

    FIXED = "fixed"
    LEARNED = "learned"


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
