"""Tests for api."""
from unittest.mock import MagicMock, patch

from sotai.api import (
    get_inference_results,
    get_inference_status,
    post_inference,
    post_pipeline,
    post_pipeline_config,
    post_pipeline_feature_configs,
    post_trained_model,
    post_trained_model_analysis,
)
from sotai.constants import SOTAI_API_ENDPOINT, SOTAI_BASE_URL

from .utils import MockResponse


@patch("requests.post", return_value=MockResponse({"uuid": "test_uuid"}))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_pipeline(mock_get_api_key, mock_post, fixture_pipeline):
    """Tests that a pipeline is posted correctly.""" ""
    pipeline_response = post_pipeline(fixture_pipeline)

    mock_post.assert_called_with(
        f"{SOTAI_BASE_URL}/{SOTAI_API_ENDPOINT}/pipelines",
        json={
            "name": "target_classification",
            "target": "target",
            "target_column_type": "classification",
            "primary_metric": "auc",
        },
        headers={"sotai-api-key": "test_api_key"},
        timeout=10,
    )

    mock_get_api_key.assert_called_once()
    assert pipeline_response[1] == "test_uuid"
    assert mock_post.call_count == 1


@patch("requests.post", return_value=MockResponse({"uuid": "test_uuid"}))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_pipeline_config(mock_get_api_key, mock_post, fixture_pipeline_config):
    """Tests that a pipeline config is posted correctly."""
    pipeline_config_response = post_pipeline_config(
        "test_uuid", fixture_pipeline_config
    )

    mock_post.assert_called_with(
        f"{SOTAI_BASE_URL}/{SOTAI_API_ENDPOINT}/pipelines/test_uuid/pipeline-configs",
        json={
            "shuffle_data": False,
            "drop_empty_percentage": 80,
            "train_percentage": 60,
            "validation_percentage": 20,
            "test_percentage": 20,
        },
        headers={"sotai-api-key": "test_api_key"},
        timeout=10,
    )
    mock_get_api_key.assert_called_once()
    assert pipeline_config_response[1] == "test_uuid"


@patch("requests.post", return_value=MockResponse({"uuid": "test_uuid"}))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_feature_configs(
    mock_get_api_key,
    mock_post,
    fixture_pipeline_config,
    fixture_categories_strs,
    fixture_categories_ints,
):
    """Tests that feature configs are posted correctly."""
    pipeline_config_response = post_pipeline_feature_configs(
        "test_uuid", fixture_pipeline_config.feature_configs
    )

    mock_post.assert_called_with(
        f"{SOTAI_BASE_URL}/{SOTAI_API_ENDPOINT}/pipeline-configs/test_uuid/feature-configs",
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
                "feature_name": "categorical_strs",
                "feature_type": "categorical",
                "categories_str": fixture_categories_strs,
            },
            {
                "feature_name": "categorical_ints",
                "feature_type": "categorical",
                "categories_int": fixture_categories_ints,
            },
        ],
        headers={"sotai-api-key": "test_api_key"},
        timeout=10,
    )
    mock_get_api_key.assert_called_once()

    assert pipeline_config_response == "success"


@patch("requests.post", return_value=MockResponse({"uuid": "test_uuid"}))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_trained_model_analysis(
    mock_get_api_key, mock_post, fixture_trained_model
):
    """Tests that a trained model is posted correctly."""

    post_trained_model_analysis("test_uuid", fixture_trained_model)

    mock_post.assert_called_with(
        f"{SOTAI_BASE_URL}/{SOTAI_API_ENDPOINT}/pipeline-configs/test_uuid/analysis",
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
                "test_primary_metric": 1,
                "validation_primary_metric": [3],
                "train_primary_metric": [3],
            },
        },
        headers={"sotai-api-key": "test_api_key"},
        timeout=10,
    )
    mock_get_api_key.assert_called_once()


@patch("tarfile.open")
@patch("builtins.open")
@patch("requests.post", return_value=MockResponse({"uuid": "test_uuid"}))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_trained_model(
    mock_get_api_key, mock_post, mock_open_data, mock_tarfile_open
):
    """Tests that feature configs are posted correctly."""
    mock_add = MagicMock()
    mock_open_data.return_value.__enter__.return_value = "data"
    mock_tarfile_open.return_value.__enter__.return_value.add = mock_add
    pipeline_response = post_trained_model("/tmp/model", "test_uuid")

    mock_post.assert_called_with(
        f"{SOTAI_BASE_URL}/{SOTAI_API_ENDPOINT}/models",
        files={"file": "data"},
        data={"trained_model_uuid": "test_uuid"},
        headers={"sotai-api-key": "test_api_key"},
        timeout=10,
    )
    mock_get_api_key.assert_called_once()

    assert pipeline_response == "success"


@patch("builtins.open")
@patch(
    "requests.post",
    return_value=MockResponse({"inferenceConfigUUID": "test_inference_uuid"}),
)
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_inferencel(
    mock_get_api_key,
    mock_post,
    mock_open_data,
):
    """Tests that feature configs are posted correctly."""
    mock_open_data.return_value.__enter__.return_value = "data"
    pipeline_response = post_inference("/tmp/model", "test_uuid")

    mock_post.assert_called_with(
        f"{SOTAI_BASE_URL}/{SOTAI_API_ENDPOINT}/inferences",
        files={"file": "data"},
        data={"trained_model_uuid": "test_uuid"},
        headers={"sotai-api-key": "test_api_key"},
        timeout=10,
    )
    mock_get_api_key.assert_called_once()

    assert pipeline_response[1] == "test_inference_uuid"
    assert pipeline_response[0] == "success"


@patch("requests.get", return_value=MockResponse("initializing", 200))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_get_inference_status(mock_get_api_key, mock_get):
    """Tests that inference config retrieval is handled correctly."""
    inference_status = get_inference_status("test_uuid")

    mock_get.assert_called_with(
        f"{SOTAI_BASE_URL}/{SOTAI_API_ENDPOINT}/inferences/test_uuid/status",
        headers={"sotai-api-key": "test_api_key"},
        timeout=10,
    )

    assert inference_status[1] == "initializing"
    assert inference_status[0] == "success"
    mock_get_api_key.assert_called_once()


@patch("urllib.request.urlretrieve", return_value=None)
@patch("requests.get", return_value=MockResponse("test.com", 200))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_get_inference_result(mock_get_api_key, mock_get, mock_urlretrieve):
    """Tests that inference file retrieval is handled correctly."""
    inference_status = get_inference_results("test_uuid", "/tmp")

    mock_get.assert_called_with(
        f"{SOTAI_BASE_URL}/{SOTAI_API_ENDPOINT}/inferences/test_uuid/download",
        headers={"sotai-api-key": "test_api_key"},
        timeout=10,
    )

    assert inference_status == "success"
    mock_get_api_key.assert_called_once()
    mock_urlretrieve.assert_called_with("test.com", "/tmp/inference_results.csv")
