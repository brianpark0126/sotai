"""Pydantic models for Pipelines."""
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import pytorch_calibrated as ptcm
import tensorflow_lattice as tfl
from pydantic import BaseModel

from ..enums import (
    EnsembleType,
    FeatureType,
    InputKeypointsInit,
    InputKeypointsType,
    Interpolation,
    LossType,
    Metric,
    ModelFramework,
    ModelType,
    Monotonicity,
    Parameterization,
    TargetType,
    TransformationType,
)

# TODO (will): write better Google-style docstrings + any other necessary comments.


class CleaningConfig(BaseModel):
    """Configuration for cleaning data."""

    # Maps column names to conversion functions. Functions will be applied to the DF
    # e.g. data[column_name] = data[column_name].apply(column_converters[column_name])
    column_converters: Optional[Dict[str, Callable[[any], any]]] = None
    # Drop rows that have drop_empty_percentage or more column values missing.
    drop_empty_percentage = 70


class _FeatureConfig(BaseModel):
    """Base class for feature configs."""

    name: str
    type: FeatureType
    missing_input_value: Optional[float] = None


class NumericalFeatureConfig(_FeatureConfig):
    """Configuration for a numerical feature."""

    num_keypoints: int = 10
    input_keypoints_init: InputKeypointsInit = InputKeypointsInit.QUANTILES
    input_keypoints_type: InputKeypointsType = InputKeypointsType.FIXED
    monotonicity: Monotonicity = Monotonicity.NONE


class CategoricalFeatureConfig(_FeatureConfig):
    """Configuration for a categorical feature."""

    categories: Union[List[str], List[int]]
    monotonicity_pairs: Optional[List[Tuple[str, str]]] = None


class TransformationConfig(NumericalFeatureConfig):
    """Configuration for a transformation feature."""

    transformation_type: TransformationType
    # Must be the name of a column in the dataset.
    primary_feature: str
    # For ADD and MULTIPLY, must provide at least one of secondary feature / value.
    secondary_feature: Optional[str] = None
    secondary_value: Optional[float] = None


class _ModelOptions(BaseModel):
    """Base class for model options."""

    output_min: Optional[float] = None
    output_max: Optional[float] = None
    output_calibration: bool = False
    output_calibration_num_keypoints: int = 10
    output_initialization: InputKeypointsInit = InputKeypointsInit.QUANTILES
    output_calibration_input_keypoints_type: InputKeypointsType = (
        InputKeypointsType.FIXED
    )


class LinearOptions(_ModelOptions):
    """Calibrated Linear model options."""

    use_bias: bool = True


class LatticeOptions(_ModelOptions):
    """Calibrated Lattice model options."""

    lattice_size: int = 2
    interpolation: Interpolation = Interpolation.KFL
    parameterization: Parameterization = Parameterization.HYPERCUBE
    num_terms: int = 2
    random_seed: int = 42


class EnsembleOptions(LatticeOptions):
    """Calibrated Lattice Ensemble model options."""

    lattices: EnsembleType = EnsembleType.RANDOM
    # Note: num_lattices * lattice_rank should be >= num_features
    # This can be properly defaulted in the configure function.
    num_lattices: Optional[int] = None
    lattice_rank: Optional[int] = None
    separate_calibrators: bool = True
    use_linear_combination: bool = False
    use_bias: bool = False
    fix_ensemble_for_2d_contraints: bool = True


class ModelConfig(BaseModel):
    """Configuration for a calibrated model."""

    framework: ModelFramework
    type: ModelType
    options: Union[LinearOptions, LatticeOptions, EnsembleOptions]


class TrainingConfig(BaseModel):
    """Configuration for training a single model."""

    loss_type: Optional[LossType] = None
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3


class HypertuneConfig(BaseModel):
    """Configuration for hyperparameter tuning to find the best model."""

    epochs_options: List[int] = [50, 100]
    batch_size_options: List[int] = [32, 64]
    Linear_rate_options: List[float] = [1e-3, 1e-4]


class FeatureAnalysis(BaseModel):
    """Feature analysis results for a single feature of a trained model."""

    feature_name: str
    feature_type: str
    min: float
    max: float
    mean: float
    median: float
    std: float
    # One of the keypoint inputs must exist, which one depends on feature_type.
    keypoints_inputs_numerical: Optional[List[float]]
    keypoints_inputs_categorical: Optional[List[str]]
    keypoints_outputs: List[float]


class TrainingResults(BaseModel):
    """Training results for a single calibrated model."""

    training_time: float
    evaluation_time: float
    feature_analysis_extraction_time: float
    train_loss_by_epoch: List[float]
    train_primary_metric_by_epoch: List[float]
    validation_loss_by_epoch: List[float]
    validation_primary_metric_by_epoch: List[float]
    test_loss: float
    test_primary_metric: float
    feature_analysis_objects: Dict[str, FeatureAnalysis]
    feature_importances: Dict[str, float]


class Model(BaseModel):
    """A calibrated model container for configs, results, and the model itself."""

    id: int
    model_config: ModelConfig
    training_config: TrainingConfig
    training_results: TrainingResults
    model: Union[
        tfl.premade.CalibratedLinear,
        tfl.premade.CalibratedLattice,
        tfl.premade.CalibratedLatticeEnsemble,
        ptcm.models.CalibratedLinear,
    ]


class PipelineModels(BaseModel):
    """A container for the best model / metric and all models trained in a pipeline."""

    best_model_id: int
    best_primary_metric: float
    models: Dict[int, Model]


class PipelineConfig(BaseModel):
    """A configuration object for a `Pipeline`."""

    id: int
    columns: List[str]
    target: str
    target_type: TargetType
    primary_metric: Metric
    cleaning_config: Optional[CleaningConfig] = None
    features: Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]]
    transformations: Optional[Dict[str, TransformationConfig]] = None


class DatasetSplit(BaseModel):
    """Defines the split percentage for train, val, and test datasets."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class PreparedData(BaseModel):
    """A train, val, and test set of data that's been cleaned and transformed."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class Data(BaseModel):
    """A class for managing data."""

    id: int
    raw_data: pd.DataFrame
    dataset_split: DatasetSplit
    prepared_data: PreparedData


class PipelineData(BaseModel):
    """A class for managing pipeline data."""

    current_data_id: Optional[int] = None
    data: Dict[int, Data]
