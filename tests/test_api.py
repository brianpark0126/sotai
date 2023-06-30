from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from requests import Response

from sotai import CategoricalFeatureConfig, NumericalFeatureConfig, Pipeline, TargetType
from sotai.api import (
    post_pipeline,
    post_pipeline_config,
    post_pipeline_feature_configs,
    post_trained_model_analysis,
)
from sotai.constants import SOTAI_API_ENDPOINT
from sotai.types import PipelineConfig

from .fixtures import (
    fixture_test_categories,
    fixture_test_data,
    fixture_test_feature_configs,
    fixture_test_feature_names,
    fixture_test_pipeline,
    fixture_test_pipeline_config,
    fixture_test_target,
    fixture_test_trained_model,
)
from .utils import construct_trained_model


class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data


@patch("requests.post", return_value=MockResponse({"uuid": "test_uuid"}))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_pipeline(
    mock_get_api_key, mock_post, test_pipeline: fixture_test_pipeline
):
    """Tests that a pipeline is posted correctly.""" ""
    pipeline_uuid = post_pipeline(test_pipeline)

    mock_post.assert_called_with(
        f"{SOTAI_API_ENDPOINT}/api/v1/pipelines",
        json={
            "name": "target_classification",
            "target": "target",
            "target_column_type": "classification",
            "primary_metric": "auc",
        },
        headers={"sotai-api-key": "test_api_key"},
    )

    assert pipeline_uuid == "test_uuid"
    assert mock_post.call_count == 1


@patch("requests.post", return_value=MockResponse({"uuid": "test_uuid"}))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_pipeline_config(
    mock_get_api_key, mock_post, test_pipeline_config: fixture_test_pipeline_config
):
    """Tests that a pipeline config is posted correctly."""
    pipeline_config_uuid = post_pipeline_config("test_uuid", test_pipeline_config)

    mock_post.assert_called_with(
        f"{SOTAI_API_ENDPOINT}/api/v1/pipelines/test_uuid/pipeline-configs",
        json={
            "shuffle_data": False,
            "drop_empty_percentage": 80,
            "train_percentage": 60,
            "validation_percentage": 20,
            "test_percentage": 20,
        },
        headers={"sotai-api-key": "test_api_key"},
    )

    assert pipeline_config_uuid == "test_uuid"


@patch("requests.post", return_value=MockResponse({"uuid": "test_uuid"}))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_feature_configs(
    mock_get_api_key, mock_post, test_pipeline_config: fixture_test_pipeline_config
):

    pipeline_config_id = post_pipeline_feature_configs(
        "test_uuid", test_pipeline_config.feature_configs
    )

    mock_post.assert_called_with(
        f"{SOTAI_API_ENDPOINT}/api/v1/pipeline-configs/test_uuid/feature-configs",
        json=[
            {
                "feature_name": "numerical",
                "feature_type": "numerical",
                "num_keypoints": 10,
                "monotonicity": "increasing",
                "input_keypoints_init": "quantiles",
                "input_keypoints_type": "fixed",
            },
            {
                "feature_name": "categorical",
                "feature_type": "categorical",
                "categories_str": ["a", "b", "c", "d"],
            },
        ],
        headers={"sotai-api-key": "test_api_key"},
    )

    assert pipeline_config_id == "test_uuid"


@patch("requests.post", return_value=MockResponse({"uuid": "test_uuid"}))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_trained_model(
    mock_get_api_key, mock_post, test_trained_model: fixture_test_trained_model
):
    """Tests that a trained model is posted correctly."""

    post_trained_model_analysis("test_uuid", test_trained_model)

    mock_post.assert_called_with(
        f"{SOTAI_API_ENDPOINT}/api/v1/pipeline-configs/test_uuid/analysis",
        json={
            "feature_analyses": [
                {
                    "feature_name": "test",
                    "feature_type": "numerical",
                    "keypoints_inputs_categorical": None,
                    "keypoints_inputs_numerical": [1.0, 2.0, 3.0],
                    "keypoints_outputs": [1.0, 2.0, 3.0],
                    "statistic_max": 2.0,
                    "statistic_mean": 3.0,
                    "statistic_median": 4.0,
                    "statistic_min": 1.0,
                    "statistic_std": 5.0,
                }
            ],
            "model_config": {
                "loss_type": "mse",
                "model_config_name": "Model 1",
                "model_framework": "pytorch",
                "model_type": "linear",
                "primary_metric": "auc",
                "target_column": "target",
                "target_column_type": "classification",
            },
            "overall_model_results": {
                "batch_size": 32,
                "epochs": 100,
                "feature_names": ["test"],
                "learning_rate": 0.001,
                "linear_coefficients": [1.0],
                "runtime_in_seconds": 1.0,
                "test_loss": 1.0,
                "test_primary_metric": 1.0,
                "train_loss_per_epoch": [1.0, 2.0, 3.0],
                "train_primary_metric_per_epoch": [1.0, 2.0, 3.0],
                "validation_loss_per_epoch": [1.0, 2.0, 3.0],
                "validation_primary_metric_per_epoch": [1.0, 2.0, 3.0],
            },
            "trained_model_metadata": {
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
            },
        },
        headers={"sotai-api-key": "test_api_key"},
    )
