"""PyTorch Calibrated training utility functions."""

from ..enums import Metric, TargetType
from ..types import (
    Dataset,
    ModelConfig,
    PipelineConfig,
    TrainedModel,
    TrainingConfig,
    TrainingResults,
)


def train_and_evaluate_ptcm_model(
    dataset_id: int,
    dataset: Dataset,
    target: str,
    target_type: TargetType,
    primary_metric: Metric,
    pipeline_config_id: int,
    pipeline_config: PipelineConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> TrainedModel:
    """Trains a PyTorch Calibrated model according to the given config."""
    # 1. Convert data frames to CSVData
    # 2. Create PTCM feature configs.
    # 3. Configure PTCM model config.
    # 4. Train PTCM model with Adam optimizer and evaluate with primary metric.
    # 5. Return trained model with training results.
    raise NotImplementedError()
