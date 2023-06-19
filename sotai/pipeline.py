"""A Pipeline for calibrated modeling."""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .enums import Metric, TargetType
from .types import (
    Dataset,
    HypertuneConfig,
    ModelConfig,
    PipelineConfig,
    PipelineRun,
    PreparedData,
    TrainedModel,
    TrainingConfig,
)
from .utils import default_feature_configs


class Pipeline:  # pylint: disable=too-many-instance-attributes
    """A pipeline for calibrated modeling.

    A pipline takes in raw data and outputs a calibrated model. This process breaks
    down into the following steps:

    - Preparation. The data is prepared and split into train, val, and test sets.
    - Training. Models are trained on the train and val sets.

    You can then analyze trained models and their results, and you can use the best
    model that you trust to make predictions on new data.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data: pd.DataFrame,
        target: str,
        target_type: Optional[TargetType] = None,
        primary_metric: Optional[Metric] = None,
        name: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ):
        """Initializes an instance of `Pipeline`.

        The pipeline is initialized with a default config, which can be modified later.
        The target type can be optionally specfified. If not specified, the pipeline
        will try to automatically determine the type of the target from the data. The
        same is true for the primary metric. The default primary metric will be F1 score
        for classification and Mean Squared Error for regression.

        Args:
            data: The raw data to be used for training.
            target: The name of the target column.
            target_type: The type of the target column.
            primary_metric: The primary metric to use for training and evaluation.
            name: The name of the pipeline. If not provided, the name will be set to
                `{target}_{target_type}`.
            categories: The column names in `data` for categorical columns.
        """
        self.target: str = target
        self.target_type: TargetType = (
            target_type
            if target_type is not None
            else self._determine_target_type(data[self.target])
        )
        self.primary_metric: Metric = (
            primary_metric
            if primary_metric is not None
            else (
                Metric.F1
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
            features=default_feature_configs(data, self.target, categories),
        )

    def prepare(
        self,
        data: Optional[pd.DataFrame] = None,
        pipeline_config_id: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Prepares the pipeline and dataset for training given the preparation config.

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

        dataset = Dataset(
            pipeline_config_id=pipeline_config_id,
            columns=data.columns.to_list(),
            prepared_data=PreparedData(train=train_data, val=val_data, test=test_data),
        )
        dataset_id = len(self.datasets) + 1
        self.datasets[dataset_id] = dataset

        return dataset_id, pipeline_config_id

    def train(
        self,
        dataset_id: int,
        pipeline_config_id: int,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> TrainedModel:
        """Returns a model trained according to the model and training configs."""
        raise NotImplementedError()

    def hypertune(
        self,
        model_config: ModelConfig,
        hypertune_config: HypertuneConfig,
        dataset: Optional[Union[Dataset, int]] = None,
        pipeline_config: Optional[Union[PipelineConfig, int]] = None,
    ) -> Tuple[int, float, List[int]]:
        """Runs hyperparameter tuning for the pipeline according to the given config.

        Args:
            dataset_id: The id of the dataset to be used for training.
            pipeline_config_id: The id of the pipeline config to be used for training.
            model_config: The config for the model to be trained.
            hypertune_config: The config for hyperparameter tuning.

        Returns:
            A tuple of the best model id, the best model's primary metric, and a list of
            all model ids that were trained.
        """
        raise NotImplementedError()

    def run(  # pylint: disable=too-many-arguments
        self,
        dataset: Optional[Union[pd.DataFrame, int]] = None,
        pipeline_config_id: Optional[int] = None,
        model_config: Optional[ModelConfig] = None,
        hypertune_config: Optional[HypertuneConfig] = None,
    ) -> PipelineRun:
        """Runs the pipeline according to the pipeline and training configs.

        The full pipeline run process is as follows:
            - Prepare the data.
            - Train the model.

        In future versions, running the pipeline will also include hyperparameter tuning
        to make it easier to find the best performing model.

        When `data` is not specified, the pipeline will use the most recently used data
        unless this is the first run, in which case it will use the data that was passed
        in during initialization. When `model_config` is not specified, the pipeline will
        use the default model config. When `hypertune_config` is not specified, the
        pipeline will use the default hypertune config.

        A call to `run` will create new dataset and pipeline config versions unless
        explicit ids for previous versions are provided.

        Args:
            dataset: The data to be used for training. Can be a pandas DataFrame
                containing new data or the id of a previously used dataset. If not
                specified, the pipeline will use the most recently used dataset unless
                this is the first run, in which case it will use the data that was
                passed in during initialization.
            pipeline_config_id: The id of the pipeline config to be used for training.
                If not specified, the pipeline will use the current settings for the
                primary pipeline config.
            model_config: The config for the model to be trained.
            hypertune_config: The config for hyperparameter tuning.

        Returns:
            A tuple of the best model id, the best model's primary metric, and a list of
            all model ids that were trained.
        """
        raise NotImplementedError()

    def predict(
        self, data: pd.DataFrame, model_id: int = None
    ) -> Tuple[pd.DataFrame, str]:
        """Runs pipeline without training to generate predictions for given data.

        Args:
            data: The data to be used for prediction. Must have all columns used for
                training the model to be used.
            model_id: The id of the model to be used for prediction.

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

    #############################
    #     Private Functions     #
    #############################

    def _determine_target_type(self, target_data: pd.Series) -> TargetType:
        """Returns the type of a target determined from its data."""
        if target_data.dtype.kind in ["i", "u"] and sorted(target_data.unique()) == [
            0,
            1,
        ]:
            return TargetType.CLASSIFICATION
        if target_data.dtype.kind == "f":
            return TargetType.REGRESSION
        return TargetType.UNKNOWN
