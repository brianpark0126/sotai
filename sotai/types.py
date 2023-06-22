"""Pydantic models for Pipelines."""
from typing import Dict, List, Optional, Union

import pandas as pd
import pytorch_calibrated as ptcm
import tensorflow_lattice as tfl
from pydantic import BaseModel, Field, root_validator

from .enums import (
    FeatureType,
    InputKeypointsInit,
    InputKeypointsType,
    LossType,
    ModelFramework,
    Monotonicity,
)


class DatasetSplit(BaseModel):
    """Defines the split percentage for train, val, and test datasets.

    Attributes:
        train: The percentage of the dataset to use for training.
        val: The percentage of the dataset to use for validation.
        test: The percentage of the dataset to use for testing.
    """

    train: int = 80
    val: int = 10
    test: int = 10

    @root_validator(pre=True, allow_reuse=True)
    @classmethod
    def validate_split_sum(cls, values):
        """Ensures that the split percentages add up to 100."""
        assert (
            values["train"] + values["val"] + values["test"] == 100
        ), "split percentages must add up to 100"
        return values


class PreparedData(BaseModel):
    """A train, val, and test set of data that's been cleaned.

    Attributes:
        train: The training dataset.
        val: The validation dataset.
        test: The testing dataset.
    """

    train: pd.DataFrame = Field(...)
    val: pd.DataFrame = Field(...)
    test: pd.DataFrame = Field(...)

    class Config:  # pylint: disable=missing-class-docstring,too-few-public-methods
        arbitrary_types_allowed = True


class Dataset(BaseModel):
    """A class for managing data.

    Attributes:
        pipeline_config_id: The ID of the pipeline config used to create this dataset.
        prepared_data: The prepared data ready for training.
    """

    pipeline_config_id: int = Field(...)
    prepared_data: PreparedData = Field(...)

    class Config:  # pylint: disable=missing-class-docstring,too-few-public-methods
        arbitrary_types_allowed = True


class NumericalFeatureConfig(BaseModel):
    """Configuration for a numerical feature.

    Attributes:
        name: The name of the feature.
        type: The type of the feature. Always `FeatureType.NUMERICAL`.
        num_keypoints: The number of keypoints to use for the calibrator.
        input_keypoints_init: The method for initializing the input keypoints.
        input_keypoints_type: The type of input keypoints.
        monotonicity: The monotonicity constraint, if any.
    """

    name: str = Field(...)
    type: FeatureType = Field(FeatureType.NUMERICAL, const=True)
    num_keypoints: int = 10
    input_keypoints_init: InputKeypointsInit = InputKeypointsInit.QUANTILES
    input_keypoints_type: InputKeypointsType = InputKeypointsType.FIXED
    monotonicity: Monotonicity = Monotonicity.NONE


class CategoricalFeatureConfig(BaseModel):
    """Configuration for a categorical feature.

    Attributes:
        name: The name of the feature.
        type: The type of the feature. Always `FeatureType.CATEGORICAL`.
        categories: The categories for the feature.
    """

    name: str = Field(...)
    type: FeatureType = Field(FeatureType.CATEGORICAL, const=True)
    categories: Union[List[str], List[int]] = Field(...)
    # TODO (will): add support for categorical monotonicity.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


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


class ModelConfig(BaseModel):
    """Configuration for a calibrated model.

    Attributes:
        framework: The framework to use for the model (TensorFlow / PyTorch).
        type: The type of model to use.
        options: The configuration options for the model.
    """

    framework: ModelFramework = Field(ModelFramework.TENSORFLOW, const=True)
    # TODO (will): Add support for Calibrated Lattice and Calibrated Lattice Ensemble.
    options: LinearOptions = Field(...)


class TrainingConfig(BaseModel):
    """Configuration for training a single model.

    Attributes:
        loss_type: The type of loss function to use for training.
        epochs: The number of iterations through the dataset during training.
        batch_size: The number of examples to use for each training step.
        learning_rate: The learning rate to use for the optimizer.
    """

    loss_type: LossType = Field(...)
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3


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

    feature_name: str = Field(...)
    feature_type: FeatureType = Field(...)
    min: float = Field(...)
    max: float = Field(...)
    mean: float = Field(...)
    median: float = Field(...)
    std: float = Field(...)
    # One of the keypoint inputs must exist, which one depends on feature_type.
    keypoints_inputs_numerical: Optional[List[float]] = Field(...)
    keypoints_inputs_categorical: Optional[List[str]] = Field(...)
    keypoints_outputs: List[float] = Field(...)


class TrainingResults(BaseModel):
    """Training results for a single calibrated model.

    Attributes:
        training_time: The total time spent training the model.
        train_loss_by_epoch: The training loss for each epoch.
        train_primary_metric_by_epoch: The training primary metric for each epoch.
        val_loss_by_epoch: The validation loss for each epoch.
        val_primary_metric_by_epoch: The validation primary metric for each
            epoch.
        evaluation_time: The total time spent evaluating the model.
        test_loss: The test loss.
        test_primary_metric: The test primary metric.
        feature_analysis_extraction_time: The total time spent extracting feature
            analysis data from the model.
        feature_analyses: The feature analysis results for each feature.
        feature_importance_extraction_time: The total time spent extracting feature
            importance data from the model.
        feature_importances: The feature importances for each feature.
    """

    training_time: float = Field(...)
    train_loss_by_epoch: List[float] = Field(...)
    train_primary_metric_by_epoch: List[float] = Field(...)
    val_loss_by_epoch: List[float] = Field(...)
    val_primary_metric_by_epoch: List[float] = Field(...)
    evaluation_time: float = Field(...)
    test_loss: float = Field(...)
    test_primary_metric: float = Field(...)
    feature_analyses_extraction_time: float = Field(...)
    feature_analyses: Dict[str, FeatureAnalysis] = Field(...)
    feature_importance_extraction_time: float = Field(...)
    feature_importances: Dict[str, float] = Field(...)


class TrainedModel(BaseModel):
    """A calibrated model container for configs, results, and the model itself.

    Attributes:
        model_config: The configuration for the model.
        training_config: The configuration used for training the model.
        training_results: The results of training the model.
        model: The trained model.
    """

    dataset_id: int = Field(...)
    pipeline_config_id: int = Field(...)
    model_config: ModelConfig = Field(...)
    training_config: TrainingConfig = Field(...)
    training_results: TrainingResults = Field(...)
    model: Union[
        tfl.premade.CalibratedLinear,
        tfl.premade.CalibratedLattice,
        tfl.premade.CalibratedLatticeEnsemble,
        ptcm.models.CalibratedLinear,
    ] = Field(...)

    class Config:  # pylint: disable=missing-class-docstring,too-few-public-methods
        arbitrary_types_allowed = True


class PipelineConfig(BaseModel):
    """A configuration object for a `Pipeline`.

    Attributes:
        shuffle_data: Whether to shuffle the data before splitting it into train,
            validation, and test sets.
        drop_empty_percentage: Rows will be dropped if they are this percentage empty.
        dataset_split: The split of the dataset into train, validation, and test sets.
        features: A dictionary mapping the column name for a feature to its config.
    """

    shuffle_data: bool = True
    drop_empty_percentage: int = 70
    dataset_split: DatasetSplit = DatasetSplit(train=80, val=10, test=10)
    features: Dict[
        str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]
    ] = Field(...)
