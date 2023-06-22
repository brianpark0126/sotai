"""A Pipeline for calibrated modeling."""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .constants import MISSING_CATEGORY_VALUE, MISSING_NUMERICAL_VALUE
from .enums import FeatureType, LossType, Metric, ModelFramework, TargetType
from .modeling_utils.pytorch_calibrated import (
    ptcm_model_predict,
    train_and_evaluate_ptcm_model,
)
from .modeling_utils.tensorflow_lattice import (
    tfl_model_predict,
    train_and_evaluate_tfl_model,
)
from .types import (
    CategoricalFeatureConfig,
    Dataset,
    LinearOptions,
    ModelConfig,
    NumericalFeatureConfig,
    PipelineConfig,
    PreparedData,
    TrainedModel,
    TrainingConfig,
)
from .utils import determine_feature_types


class Pipeline:  # pylint: disable=too-many-instance-attributes
    """A pipeline for calibrated modeling.

    A pipline takes in raw data and outputs a calibrated model. This process breaks
    down into the following steps:

    - Preparation. The data is prepared and split into train, val, and test sets.
    - Training. Models are trained on the train and val sets.

    You can then analyze trained models and their results, and you can use the best
    model that you trust to make predictions on new data.
    """

    def __init__(
        self,
        features: List[str],
        target: str,
        target_type: TargetType,
        categories: Optional[Dict[str, List[str]]] = None,
        primary_metric: Optional[Metric] = None,
        name: Optional[str] = None,
    ):
        """Initializes an instance of `Pipeline`.

        The pipeline is initialized with a default config, which can be modified later.
        The target type can be optionally specfified. The default primary metric will be
        AUC score for classification and Mean Squared Error for regression if not
        specified.

        Args:
            features: The column names in your data to use as features.
            target: The name of the target column.
            target_type: The type of the target column.
            categories: A dictionary mapping feature names to unique categories. Any
                values not in the categories list for a given feature will be treated
                as a missing value.
            primary_metric: The primary metric to use for training and evaluation.
            name: The name of the pipeline. If not provided, the name will be set to
                `{target}_{target_type}`.
        """
        self.target: str = target
        self.target_type: TargetType = target_type
        self.primary_metric: Metric = (
            primary_metric
            if primary_metric is not None
            else (
                Metric.AUC
                if self.target_type == TargetType.CLASSIFICATION
                else Metric.MSE
            )
        )
        # Maps a PipelineConfig id to its corresponding PipelineConfig instance.
        self.configs: Dict[int, PipelineConfig] = {}
        # Maps a Dataset id to its corresponding Dataset instance.
        self.datasets: Dict[int, Dataset] = {}
        # Maps a TrainedModel id to its corresponding TrainedModel instance.
        self.models: Dict[int, TrainedModel] = {}

        self.name: str = name if name else f"{self.target}_{self.target_type}"
        self.config: PipelineConfig = PipelineConfig(
            features={
                feature_name: (
                    CategoricalFeatureConfig(
                        name=feature_name,
                        categories=categories[feature_name],
                    )
                    if categories and feature_name in categories
                    else NumericalFeatureConfig(name=feature_name)
                )
                for feature_name in features
            },
        )

    def prepare(  # pylint: disable=too-many-locals
        self,
        data: pd.DataFrame,
        pipeline_config_id: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Prepares the pipeline and dataset for training given the preparation config.

        If any features in data are detected as non-numeric, the pipeline will attempt
        to handle them as categorical features. Any features that the pipeline cannot
        handle will be skipped.

        Args:
            data: The raw data to be prepared for training.
            pipeline_config_id: The id of the pipeline config to be used for training.
                If not provided, the current pipeline config will be used and versioned.

        Returns:
            A tuple of the dataset id and the pipeline config id used for preparation.
        """
        if pipeline_config_id is None:
            pipeline_config_id = len(self.configs) + 1
            pipeline_config = copy.deepcopy(self.config)
            self.configs[pipeline_config_id] = pipeline_config
        else:
            pipeline_config = self.configs[pipeline_config_id]

        feature_types = determine_feature_types(
            data[list(pipeline_config.features.keys())]
        )
        for feature_name, feature_type in feature_types.items():
            feature_config = pipeline_config.features[feature_name]
            if (
                feature_type == FeatureType.NUMERICAL
                or feature_config.type == FeatureType.CATEGORICAL
            ):
                continue
            if feature_type == FeatureType.CATEGORICAL:
                logging.info(
                    "Detected %s as categorical. Replacing numerical config with "
                    "default categorical config using all unique values as categories",
                    feature_name,
                )
                pipeline_config.features[feature_name] = CategoricalFeatureConfig(
                    name=feature_name,
                    categories=sorted(data[feature_name].unique().tolist()),
                )
            else:
                logging.info(
                    "Removing feature %s because its data type is not supported.",
                    feature_name,
                )
                pipeline_config.features.pop(feature_name)

        # Select only the features defined in the pipeline config.
        data = data[list(pipeline_config.features.keys()) + [self.target]]

        # Drop rows with too many missing values according to the drop empty percent.
        data.replace("", np.nan, inplace=True)
        max_num_empty_columns = int(
            (pipeline_config.drop_empty_percentage * data.shape[1]) / 100
        )
        data = data[data.isnull().sum(axis=1) <= max_num_empty_columns]

        # Replace any missing values (i.e. NaN) with missing value constants.
        self._replace_missing_values(data, pipeline_config)

        if pipeline_config.shuffle_data:
            data = data.sample(frac=1).reset_index(drop=True)

        train_percentage = pipeline_config.dataset_split.train / 100
        train_data = data.iloc[: int(len(data) * train_percentage)]
        val_percentage = pipeline_config.dataset_split.val / 100
        val_data = data.iloc[
            int(len(data) * train_percentage) : int(
                len(data) * (train_percentage + val_percentage)
            )
        ]
        test_data = data.iloc[int(len(data) * (train_percentage + val_percentage)) :]

        dataset_id = len(self.datasets) + 1
        dataset = Dataset(
            id=dataset_id,
            pipeline_config_id=pipeline_config_id,
            prepared_data=PreparedData(train=train_data, val=val_data, test=test_data),
        )
        self.datasets[dataset_id] = dataset

        return dataset_id, pipeline_config_id

    def train(
        self,
        dataset_id: int,
        pipeline_config_id: int,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ) -> Tuple[int, TrainedModel]:
        """Returns a calibrated model trained according to the given configs."""
        dataset = self.datasets[dataset_id]
        pipeline_config = self.configs[pipeline_config_id]
        if model_config is None:
            model_config = ModelConfig(
                framework=ModelFramework.TENSORFLOW, options=LinearOptions()
            )
        if training_config is None:
            training_config = TrainingConfig(
                loss_type=LossType.BINARY_CROSSENTROPY
                if self.target_type == TargetType.CLASSIFICATION
                else LossType.MSE
            )

        if model_config.framework == ModelFramework.TENSORFLOW:
            trained_model = train_and_evaluate_tfl_model(
                dataset_id,
                dataset,
                self.target,
                self.target_type,
                self.primary_metric,
                pipeline_config_id,
                pipeline_config,
                model_config,
                training_config,
            )
        elif model_config.framework == ModelFramework.PYTORCH:
            trained_model = train_and_evaluate_ptcm_model(
                dataset_id,
                dataset,
                self.target,
                self.primary_metric,
                pipeline_config_id,
                pipeline_config,
                model_config,
                training_config,
            )
        else:
            raise ValueError(f"Unknown model framework: {model_config.framework}.")

        model_id = len(self.models) + 1
        self.models[model_id] = trained_model
        return model_id, trained_model

    def predict(
        self, data: pd.DataFrame, model_id: int
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Runs pipeline without training to generate predictions for given data.

        Args:
            data: The data to be used for prediction. Must have all columns used for
                training the model to be used.
            model_id: The id of the model to be used for prediction.

        Returns:
            If the pipeline target type is regression, a numpy array of predictions.
            If the pipeline target type is classification, a tuple containing a numpy
            array of predictions (logits) and a numpy array of probabilities.
        """
        trained_model = self.models[model_id]
        pipeline_config = self.configs[trained_model.pipeline_config_id]
        data = data[list(pipeline_config.features.keys())]
        self._replace_missing_values(data, pipeline_config)

        if trained_model.model_config.framework == ModelFramework.TENSORFLOW:
            predictions = tfl_model_predict(
                trained_model.model, pipeline_config.features, data
            )
        elif trained_model.model_config.framework == ModelFramework.PYTORCH:
            predictions = ptcm_model_predict(trained_model.model, data)
        else:
            raise ValueError(f"Unknown model framework: {trained_model.framework}.")

        if self.target_type == TargetType.REGRESSION:
            return predictions
        elif self.target_type == TargetType.CLASSIFICATION:
            return predictions, 1.0 / (1.0 + np.exp(-predictions))
        else:
            raise ValueError(f"Unknown target type: {self.target_type}.")

    def save(self, filename: str):
        """Saves the pipeline to a file.

        Args:
            filename: The name of the file to save the pipeline to.
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, filename: str) -> Pipeline:
        """Loads the pipeline from a file.

        Args:
            filename: The name of the file to load the pipeline from.

        Returns:
            An instance of `Pipeline` loaded from the file.
        """
        raise NotImplementedError()

    ################################################################################
    #                              PRIVATE METHODS                                 #
    ################################################################################

    def _replace_missing_values(
        self, data: pd.DataFrame, pipeline_config: PipelineConfig
    ):
        """Replaces empty values or unspecified categories with a constant value."""
        for feature_name, feature_config in pipeline_config.features.items():
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
