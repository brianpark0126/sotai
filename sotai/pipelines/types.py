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
    """Configuration for cleaning data.

    Attributes:
        column_converters: Maps column names to conversion functions. Functions should
            take in a single value as input and output a single converted value. Each
            column's function will be applied to the DF e.g.
            data[column_name] = data[column_name].apply(column_converters[column_name]).
        drop_empty_percentage: Drop rows that have drop_empty_percentage or more column
            values missing.
    """

    column_converters: Optional[Dict[str, Callable[[any], any]]] = None
    drop_empty_percentage = 70


class _FeatureConfig(BaseModel):
    """Base class for feature configs.

    Attributes:
        name: The name of the feature.
        type: The type of the feature.
        missing_input_value: The value that represents a missing input value. If None,
            then it will be assumed that no values are missing for this feature.
    """

    name: str
    type: FeatureType
    missing_input_value: Optional[float] = None


class NumericalFeatureConfig(_FeatureConfig):
    """Configuration for a numerical feature.

    Attributes:
        num_keypoints: The number of keypoints to use for the calibrator.
        input_keypoints_init: The method for initializing the input keypoints.
        input_keypoints_type: The type of input keypoints.
        monotonicity: The monotonicity constraint, if any.
    """

    num_keypoints: int = 10
    input_keypoints_init: InputKeypointsInit = InputKeypointsInit.QUANTILES
    input_keypoints_type: InputKeypointsType = InputKeypointsType.FIXED
    monotonicity: Monotonicity = Monotonicity.NONE


class CategoricalFeatureConfig(_FeatureConfig):
    """Configuration for a categorical feature.

    Attributes:
        categories: The categories for the feature.
        monotonicity_pairs: The monotonicity constraints, if any, defined as pairs of
            categories. The output for the second category will be greater than or equal
            to the output for the first category for each pair, all else being equal.
    """

    categories: Union[List[str], List[int]]
    monotonicity_pairs: Optional[List[Tuple[str, str]]] = None


class TransformationConfig(NumericalFeatureConfig):
    """Configuration for a transformation feature.

    Attributes:
        transformation_type: The type of transformation.
        primary_feature: The name of the primary feature. This must match a column name
            in the dataset to be transformed.
        secondary_feature: The name of the secondary feature, if any, to use for the
            transformation. Either this or `secondary_value` must be provided for
            transformations that operate on two values.
        secondary_value: The secondary value, if any, to use for the transformation.
            Either this or `secondary_feature` must be provided for transformations that
            operate on two values.
    """

    transformation_type: TransformationType
    # Must be the name of a column in the dataset.
    primary_feature: str
    # For ADD and MULTIPLY, must provide at least one of secondary feature / value.
    secondary_feature: Optional[str] = None
    secondary_value: Optional[float] = None


class _ModelOptions(BaseModel):
    """Base class for model options.

    Attributes:
        output_min: The minimum output value for the model. If None, then it will be
            assumed that there is no minimum output value.
        output_max: The maximum output value for the model. If None, then it will be
            assumed that there is no maximum output value.
        output_calibration: Whether to calibrate the output.
        output_calibration_num_keypoints: The number of keypoints to use for the output
            calibrator.
        output_initialization: The method for initializing the output calibrator input
            keypoints.
        output_calibration_input_keypoints_type: The type of output calibrator input
            keypoints.
    """

    output_min: Optional[float] = None
    output_max: Optional[float] = None
    output_calibration: bool = False
    output_calibration_num_keypoints: int = 10
    output_initialization: InputKeypointsInit = InputKeypointsInit.QUANTILES
    output_calibration_input_keypoints_type: InputKeypointsType = (
        InputKeypointsType.FIXED
    )


class LinearOptions(_ModelOptions):
    """Calibrated Linear model options.

    Attributes:
        use_bias: Whether to use a bias term for the linear combination.
    """

    use_bias: bool = True


class LatticeOptions(_ModelOptions):
    """Calibrated Lattice model options.

    Attributes:
        lattice_size: The size of the lattice. For 1D lattices, this is the number of
            vertices. For higher-dimensional lattices, this is the number of vertices
            along each dimension. For example, a 2x2 lattice has 4 vertices.
        interpolation: The interpolation method to use for the lattice. Hypercube
            interpolation interpolates all vertices in the lattice. Simplex
            interpolation only interpolates the vertices along the edges of the lattice
            simplices.
        parameterization: The parameterization method to use for the lattice. Lattices
            with lattice size `L` and `N` inputs have ``L ** N`` parameters. All
            vertices parameterizes the lattice uses all ``L ** N`` vertices. KFL uses a
            factorized form that grows linearly with ``N``.
        num_terms: The number of terms to use for a kroncker-factored lattice. This will
            be ignored if the parameterization is not KFL.
        random_seed: The random seed to use for the lattice.
    """

    lattice_size: int = 2
    interpolation: Interpolation = Interpolation.HYPERCUBE
    parameterization: Parameterization = Parameterization.KFL
    num_terms: int = 2
    random_seed: int = 42


class EnsembleOptions(LatticeOptions):
    """Calibrated Lattice Ensemble model options.

    Attributes:
        lattices: The type of ensembling to use for lattice arrangement.
        num_lattices: The number of lattices to use for the ensemble.
        lattice_rank: The number of features to use for each lattice in the ensemble.
        separate_calibrators: Whether to use separate calibrators for each lattice in
            the ensemble. If False, then a single calibrator will be used for each input
            feature.
        use_linear_combination: Whether to use a linear combination of the lattices in
            the ensemble. If False, then the output will be the average of the outputs.
        use_bias: Whether to use a bias term for the linear combination. Ignored if
            `use_linear_combination` is False.
        fix_ensemble_for_2d_contraints: Whether to fix the ensemble arrangement for 2D
            constraints.
    """

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
    """Configuration for a calibrated model.

    Attributes:
        framework: The framework to use for the model (TensorFlow / PyTorch).
        type: The type of model to use.
        options: The configuration options for the model.
    """

    framework: ModelFramework
    type: ModelType
    options: Union[LinearOptions, LatticeOptions, EnsembleOptions]


class TrainingConfig(BaseModel):
    """Configuration for training a single model.

    Attributes:
        loss_type: The type of loss function to use for training.
        epochs: The number of iterations through the dataset during training.
        batch_size: The number of examples to use for each training step.
        learning_rate: The learning rate to use for the optimizer.
    """

    loss_type: Optional[LossType] = None
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3


class HypertuneConfig(BaseModel):
    """Configuration for hyperparameter tuning to find the best model.

    Attributes:
        epochs_options: A list of values to try for how many epochs to train the model.
        batch_size_options: A list of values to try for how many examples to use for
            each training step.
        Linear_rate_options: A list of values to try for the learning rate to use for
            the optimizer.
    """

    epochs_options: List[int] = [50, 100]
    batch_size_options: List[int] = [32, 64]
    Linear_rate_options: List[float] = [1e-3, 1e-4]


class FeatureAnalysis(BaseModel):
    """Feature analysis results for a single feature of a trained model.

    Attributes:
        feature_name: The name of the feature.
        feature_type: The type of the feature.
        min: The minimum value of the feature.
        max: The maximum value of the feature.
        mean: The mean value of the feature.
        median: The median value of the feature.
        std: The standard deviation of the feature.
        keypoints_inputs_numerical: The input keypoints for the feature if the feature
            is numerical.
        keypoints_inputs_categorical: The input keypoints for the feature if the feature
            is categorical.
        keypoints_outputs: The output keypoints for each input keypoint.
    """

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
    """Training results for a single calibrated model.

    Attributes:
        training_time: The total time spent training the model.
        evaluation_time: The total time spent evaluating the model.
        feature_analysis_extraction_time: The total time spent extracting feature
            analysis data from the model.
        train_loss_by_epoch: The training loss for each epoch.
        train_primary_metric_by_epoch: The training primary metric for each epoch.
        validation_loss_by_epoch: The validation loss for each epoch.
        validation_primary_metric_by_epoch: The validation primary metric for each
            epoch.
        test_loss: The test loss.
        test_primary_metric: The test primary metric.
        feature_analysis_objects: The feature analysis results for each feature.
        feature_importances: The feature importances for each feature.
    """

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
    """A calibrated model container for configs, results, and the model itself.

    Attributes:
        id: The ID of the model.
        model_config: The configuration for the model.
        training_config: The configuration used for training the model.
        training_results: The results of training the model.
        model: The trained model.
    """

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
    """A container for the best model / metric and all models trained in a pipeline.

    Attributes:
        best_model_id: The ID of the best model.
        best_primary_metric: The primary metric of the best model.
        models: A dictionary mapping IDs to models trained in the pipeline.
    """

    best_model_id: int
    best_primary_metric: float
    models: Dict[int, Model]


class PipelineConfig(BaseModel):
    """A configuration object for a `Pipeline`.

    Attributes:
        id: The ID of the pipeline.
        columns: The columns to use from the dataset.
        target: The target column to predict.
        target_type: The type of the target column.
        primary_metric: The primary metric to use for evaluating models and selecting
            the best one from tuning results.
        cleaning_config: The configuration to use for cleaning the dataset.
        features: A dictionary mapping the column name for a feature to its config.
        transformations: A dictionary mapping the column name for a feature
            transformation to its config.
    """

    id: int
    columns: List[str]
    target: str
    target_type: TargetType
    primary_metric: Metric
    cleaning_config: Optional[CleaningConfig] = None
    features: Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]]
    transformations: Optional[Dict[str, TransformationConfig]] = None


class DatasetSplit(BaseModel):
    """Defines the split percentage for train, val, and test datasets.

    Attributes:
        train: The percentage of the dataset to use for training.
        val: The percentage of the dataset to use for validation.
        test: The percentage of the dataset to use for testing.
    """

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class PreparedData(BaseModel):
    """A train, val, and test set of data that's been cleaned and transformed.

    Attributes:
        train: The training dataset.
        val: The validation dataset.
        test: The testing dataset.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class Dataset(BaseModel):
    """A class for managing data.

    Attributes:
        id: The ID of the data.
        raw_data: The raw data.
        dataset_split: The split percentage for train, val, and test datasets.
        prepared_data: The prepared data.
    """

    id: int
    raw_data: pd.DataFrame
    dataset_split: Optional[DatasetSplit] = None
    prepared_data: Optional[PreparedData] = None


class PipelineData(BaseModel):
    """A class for managing pipeline data.

    Attributes:
        current_data_id: The ID of the current dataset to use for the pipeline.
        data: A dictionary mapping IDs to datasets.
    """

    current_data_id: Optional[int] = None
    data: Dict[int, Dataset]
