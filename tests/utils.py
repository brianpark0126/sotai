"""Testing Utilities."""
from typing import Dict, Union

import pandas as pd
import torch

from sotai import (
    CategoricalFeatureConfig,
    DatasetSplit,
    LinearConfig,
    LossType,
    Metric,
    NumericalFeatureConfig,
    PipelineConfig,
    TargetType,
    TrainedModel,
    TrainingConfig,
    TrainingResults,
)
from sotai.data import CSVData
from sotai.training import create_features, create_model


def _batch_data(examples: torch.Tensor, labels: torch.Tensor, batch_size: int):
    """A generator that yields batches of data."""
    num_examples = examples.size()[0]
    for i in range(0, num_examples, batch_size):
        yield (
            examples[i : i + batch_size],
            labels[i : i + batch_size],
        )


def train_calibrated_module(  # pylint: disable=too-many-arguments
    calibrated_module: torch.nn.Module,  # note: must have constrain(...) function
    examples: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
):
    """Trains a calibrated module for testing purposes."""
    for _ in range(epochs):
        for batched_inputs, batched_labels in _batch_data(examples, labels, batch_size):
            optimizer.zero_grad()
            outputs = calibrated_module(batched_inputs)
            loss = loss_fn(outputs, batched_labels)
            loss.backward()
            optimizer.step()
            calibrated_module.constrain()


def construct_trained_model(
    target_type: TargetType,
    data: pd.DataFrame,
    feature_configs: Dict[str, Union[CategoricalFeatureConfig, NumericalFeatureConfig]],
):
    """Returns a `TrainedModel` instance."""
    model_config = LinearConfig(output_calibration=False)
    training_config = TrainingConfig(loss_type=LossType.MSE)
    primary_metric = Metric.MSE

    features = create_features(feature_configs, CSVData(data))
    model = create_model(features, model_config)

    return TrainedModel(
        id=0,
        dataset_id=0,
        pipeline_config=PipelineConfig(
            id=0,
            target="target",
            target_type=target_type,
            primary_metric=primary_metric,
            feature_configs=feature_configs,
            shuffle_data=False,
            drop_empty_percentage=70,
            dataset_split=DatasetSplit(train=80, val=10, test=10),
        ),
        model_config=model_config,
        training_config=training_config,
        training_results=TrainingResults(
            training_time=1,
            train_loss_by_epoch=[1],
            train_primary_metric_by_epoch=[1],
            val_loss_by_epoch=[1],
            val_primary_metric_by_epoch=[1],
            evaluation_time=1,
            test_loss=1,
            test_primary_metric=1,
            feature_analyses={},
            linear_coefficients={},
        ),
        model=model,
    )


class MockResponse:
    """Mock response class for testing."""

    def __init__(self, json_data, status_code=200):
        """Mock response for testing."""
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        """Return json data."""
        return self.json_data
