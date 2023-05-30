"""A Pipeline for calibrated modeling."""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel

from ..enums import Metric, TargetType
from .types import (
    DatasetSplit,
    HypertuneConfig,
    Model,
    ModelConfig,
    PipelineConfig,
    PipelineData,
    PipelineModels,
    PreparedData,
    TrainingConfig,
)
from .utils import (
    determine_feature_types,
    determine_target_type,
    generate_default_feature_configs,
)

# TODO (will): write better Google-style docstrings.


class Pipeline(BaseModel):
    """A pipeline for calibrated modeling.

    A pipline takes in raw data and outputs a calibrated model. This process breaks
    down into the following steps:

    - Cleaning. The raw data is cleaned according to the pipeline's cleaning config.
    - Transforming. The cleaned data is transformed according to transformation configs.
    - Preparation. The transformed data is split into train, val, and test sets.
    - Training. Hypertune models on the train and val sets to find the best one.

    You can then analyze trained models and their results, and you can use the best
    model that you trust to make predictions on new data.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: pd.DataFrame,
        target: str,
        target_type: Optional[TargetType] = None,
        name: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ):
        """Initializes an instance of `Pipeline`.

        The pipeline is initialized with a default config, which can be modified later.
        The target type can be optionally specfified. If not specified, the pipeline
        will try to automatically determine the type of the target from the data. For
        classification problems, the default metric will be F1 score. For regression
        problems, the default metric will be Mean Squared Error.

        Args:
            data: The raw data to be used for training.
            target: The name of the target column.
            target_type: The type of the target column.
            name: The name of the pipeline.
            categories: The column names in `data` for categorical columns.
        """
        self.name: str = name
        self.goal: str = ""
        target_type = determine_target_type(data[target])
        feature_types = determine_feature_types(data, target, categories)
        self.config = PipelineConfig(
            columns=data.columns,
            target=target,
            target_type=target_type,
            primary_metric=Metric.F1
            if target_type == TargetType.CLASSIFICATION
            else Metric.MSE,
            features=generate_default_feature_configs(data, target, feature_types),
        )
        # Maps a PipelineData id to its corresponding PipelineData instance.
        self.data: PipelineData = PipelineData(current_data_id=0, data={0: data})
        # Maps a PipelineModels id to its corresponding PipelineModels instance.
        self.pipeline_models: Optional[PipelineModels] = None

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns data cleaned according to the pipeline cleaning config."""
        raise NotImplementedError()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Returns data transformed according to the pipeline transformation config."""
        raise NotImplementedError()

    def prepare(self, data: pd.DataFrame, split: DatasetSplit) -> PreparedData:
        """Returns an instance of `PreparedData` for the given data and split."""
        raise NotImplementedError()

    def train(
        self,
        prepared_data: PreparedData,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> Model:
        """Returns a model trained according to the model and training configs."""
        raise NotImplementedError()

    def hypertune(
        self, prepared_data: PreparedData, hypertune_config: HypertuneConfig
    ) -> Tuple[int, float, List[int]]:
        """Runs hyperparameter tuning for the pipeline according to the given config.

        Args:
            prepared_data: The prepared data to be used for training.
            hypertune_config: The config for hyperparameter tuning.

        Returns:
            A tuple of the best model id, the best model's primary metric, and a list of
            all model ids that were trained.
        """
        raise NotImplementedError()

    def run(
        self,
        data: Optional[pd.DataFrame] = None,
        model_config: Optional[ModelConfig] = None,
        hypertune_config: Optional[HypertuneConfig] = None,
    ) -> Tuple[int, float, List[int]]:
        """Runs the pipeline according to the pipeline and training configs.

        The full pipeline run process is as follows:
            - Clean the data.
            - Transform the data.
            - Prepare the data.
            - Hypertune to find the best model.

        Args:
            data: The raw data to be used for training.
            model_config: The config for the model to be trained.
            hypertune_config: The config for hyperparameter tuning.

        Returns:
            A tuple of the best model id, the best model's primary metric, and a list of
            all model ids that were trained.
        """
        raise NotImplementedError()

    def predict(
        self, data: pd.DataFrame, model_id: Optional[int] = None
    ) -> Tuple[pd.DataFrame, str]:
        """Runs pipeline without training to generate predictions for given data.

        Args:
            data: The data to be used for prediction. Must have all columns used for
                training the model to be used.
            model_id: The id of the model to be used for prediction. If not specified,
                the best model will be used.

        Returns:
            A tuple containing a dataframe with predictions and the new of the new
            column, which will be the name of the target column with a tag appended to
            it (e.g. target_prediction).
        """
        raise NotImplementedError()

    def analyze(self, model_id: int):
        """Charts pipeline model results for a specific model.

        The following charts will be generated:
            - Calibrator charts for each feature.
            - Feature importance bar chart with feature statistics.

        Args:
            model_id: The id of the model to be analyzed.
        """
        raise NotImplementedError()

    def save(self, filanem: str):
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
