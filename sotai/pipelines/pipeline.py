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
    """A pipeline for calibrated modeling."""

    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        name: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ):
        """Initializes an instance of `Pipeline`."""
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
        """Cleans the given data according to this pipelines cleaning config."""
        raise NotImplementedError()

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Creates transformations from pipeline if not present."""
        raise NotImplementedError()

    def prepare(self, data: pd.DataFrame, split: DatasetSplit) -> PreparedData:
        """Prepares the data for model training."""
        raise NotImplementedError()

    def train(
        self,
        prepared_data: PreparedData,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> Model:
        """Trains a model on the prepared data according to the training config."""
        raise NotImplementedError()

    def hypertune(
        self, prepared_data: PreparedData, hypertune_config: HypertuneConfig
    ) -> Tuple[int, float, List[int]]:
        """Runs hyperparameter tuning on the pipeline."""
        raise NotImplementedError()

    def run(
        self,
        data: Optional[pd.DataFrame] = None,
        model_config: Optional[ModelConfig] = None,
        hypertune_config: Optional[HypertuneConfig] = None,
    ) -> Tuple[int, float, List[int]]:
        """Runs the pipeline according to the pipeline and training configs."""
        raise NotImplementedError()

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Runs pipeline without training to generate predictions for given data."""
        raise NotImplementedError()

    def analyze(self, model_id: int):
        """Charts pipeline model results for a specific model."""
        raise NotImplementedError()

    def save(self, filename: str):
        """Saves the pipeline to a file."""
        raise NotImplementedError()

    @classmethod
    def load(cls, filename: str) -> Pipeline:
        """Loads the pipeline from a file."""
        raise NotImplementedError()
