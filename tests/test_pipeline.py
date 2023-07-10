"""Tests for Pipeline."""
from unittest.mock import patch

import pytest

from sotai import (
    APIStatus,
    DatasetSplit,
    FeatureType,
    Metric,
    Pipeline,
    PipelineConfig,
    TargetType,
)
from sotai.enums import InferenceConfigStatus

from .fixtures import (  # pylint: disable=unused-import
    fixture_test_categories,
    fixture_test_data,
    fixture_test_feature_configs,
    fixture_test_feature_names,
    fixture_test_target,
)


@pytest.mark.parametrize(
    "target_type,expected_primary_metric",
    [(TargetType.CLASSIFICATION, Metric.AUC), (TargetType.REGRESSION, Metric.MSE)],
)
def test_init(
    test_feature_names: fixture_test_feature_names,
    test_target: fixture_test_target,
    target_type,
    expected_primary_metric,
):
    """Tests pipeline initialization for a classification target."""
    pipeline = Pipeline(test_feature_names, test_target, target_type)
    assert pipeline.name == f"{test_target}_{target_type}"
    assert pipeline.target == test_target
    assert pipeline.target_type == target_type
    assert pipeline.primary_metric == expected_primary_metric
    assert len(pipeline.feature_configs) == 2
    numerical_config = pipeline.feature_configs["numerical"]
    assert numerical_config.name == "numerical"
    assert numerical_config.type == FeatureType.NUMERICAL
    categorical_config = pipeline.feature_configs["categorical"]
    assert categorical_config.name == "categorical"
    # Note: we expect the default config to be numerical if not specified.
    assert categorical_config.type == FeatureType.NUMERICAL


def test_init_with_categories(
    test_feature_names: fixture_test_feature_names,
    test_target: fixture_test_target,
    test_categories: fixture_test_categories,
):
    """Tests pipeline initialization with specified categories."""
    pipeline = Pipeline(
        test_feature_names,
        test_target,
        TargetType.CLASSIFICATION,
        categories={"categorical": test_categories},
    )
    categorical_config = pipeline.feature_configs["categorical"]
    assert categorical_config.name == "categorical"
    assert categorical_config.type == FeatureType.CATEGORICAL
    assert categorical_config.categories == test_categories


@pytest.mark.parametrize(
    "target_type", [TargetType.CLASSIFICATION, TargetType.REGRESSION]
)
@pytest.mark.parametrize("metric", [Metric.AUC, Metric.MSE, Metric.MAE])
@pytest.mark.parametrize("shuffle_data", [True, False])
@pytest.mark.parametrize("drop_empty_percentage", [30, 60, 80])
@pytest.mark.parametrize(
    "dataset_split",
    [
        DatasetSplit(train=80, val=10, test=10),
        DatasetSplit(train=60, val=20, test=20),
        DatasetSplit(train=70, val=20, test=10),
    ],
)
def test_init_from_config(
    test_target: fixture_test_target,
    test_feature_configs: fixture_test_feature_configs,
    target_type,
    metric,
    shuffle_data,
    drop_empty_percentage,
    dataset_split,
):
    """Tests pipeline initialization from a `PipelineConfig` instance."""
    pipeline_config = PipelineConfig(
        id=0,
        target=test_target,
        target_type=target_type,
        primary_metric=metric,
        feature_configs=test_feature_configs,
        shuffle_data=shuffle_data,
        drop_empty_percentage=drop_empty_percentage,
        dataset_split=dataset_split,
    )
    name = "test_pipeline"
    pipeline = Pipeline.from_config(pipeline_config, name=name)
    assert pipeline.name == name
    assert pipeline.target == test_target
    assert pipeline.target_type == target_type
    assert pipeline.primary_metric == metric
    assert pipeline.feature_configs == test_feature_configs
    assert pipeline.shuffle_data == shuffle_data
    assert pipeline.drop_empty_percentage == drop_empty_percentage
    assert pipeline.dataset_split == dataset_split


def test_prepare(
    test_data: fixture_test_data,
    test_feature_names: fixture_test_feature_names,
    test_target: fixture_test_target,
    test_categories: fixture_test_target,
):
    """Tests the pipeline prepare function."""
    pipeline = Pipeline(
        test_feature_names, test_target, target_type=TargetType.CLASSIFICATION
    )
    # We set shuffle to false to ensure the data is split in the same way.
    pipeline.shuffle_data = False
    pipeline.dataset_split.train = 80
    pipeline.dataset_split.val = 10
    pipeline.dataset_split.test = 10
    dataset, pipeline_config = pipeline.prepare(test_data)
    assert pipeline_config.id == 0
    categorical_config = pipeline_config.feature_configs["categorical"]
    assert categorical_config.name == "categorical"
    assert categorical_config.type == FeatureType.CATEGORICAL
    assert categorical_config.categories == test_categories
    assert dataset.id == 0
    assert dataset.pipeline_config_id == pipeline_config.id
    num_examples = len(test_data)
    num_training_examples = int(num_examples * pipeline.dataset_split.train / 100)
    num_val_examples = int(num_examples * pipeline.dataset_split.val / 100)
    assert dataset.prepared_data.train.equals(test_data.iloc[:num_training_examples])
    assert dataset.prepared_data.val.equals(
        test_data.iloc[num_training_examples : num_training_examples + num_val_examples]
    )
    assert dataset.prepared_data.test.equals(
        test_data.iloc[num_training_examples + num_val_examples :]
    )


@pytest.mark.parametrize(
    "target_type",
    [
        (TargetType.CLASSIFICATION),
        (TargetType.REGRESSION),
    ],
)
def test_train_calibrated_linear_model(
    test_data,
    test_feature_names: fixture_test_feature_names,
    test_target: fixture_test_target,
    target_type,
):
    """Tests pipeline training for calibrated linear regression model."""
    pipeline = Pipeline(test_feature_names, test_target, target_type)
    pipeline.shuffle_data = False
    pipeline.dataset_split.train = 60
    pipeline.dataset_split.val = 20
    pipeline.dataset_split.test = 20
    trained_model = pipeline.train(test_data)
    assert len(pipeline.configs) == 1
    assert len(pipeline.datasets) == 1
    assert trained_model
    assert trained_model.dataset_id == 0
    assert pipeline.datasets[trained_model.dataset_id]
    assert trained_model.pipeline_config.id == 0
    assert pipeline.configs[trained_model.pipeline_config.id]


def test_pipeline_save_load(
    test_data: fixture_test_data,
    test_feature_names: fixture_test_feature_names,
    test_target: fixture_test_target,
    tmp_path,
):
    """Tests that an instance of `Pipeline` can be successfully saved and loaded."""
    pipeline = Pipeline(test_feature_names, test_target, TargetType.CLASSIFICATION)
    _ = pipeline.train(test_data)
    pipeline.save(tmp_path)
    loaded_pipeline = Pipeline.load(tmp_path)
    assert isinstance(loaded_pipeline, Pipeline)
    assert loaded_pipeline.name == pipeline.name
    assert loaded_pipeline.target == pipeline.target
    assert loaded_pipeline.target_type == pipeline.target_type
    assert loaded_pipeline.primary_metric == pipeline.primary_metric
    assert loaded_pipeline.feature_configs == pipeline.feature_configs
    assert loaded_pipeline.configs == pipeline.configs
    for dataset_id, loaded_dataset in loaded_pipeline.datasets.items():
        dataset = pipeline.datasets[dataset_id]
        assert loaded_dataset.id == dataset.id
        assert loaded_dataset.pipeline_config_id == dataset.pipeline_config_id
        assert loaded_dataset.prepared_data.train.equals(dataset.prepared_data.train)
        assert loaded_dataset.prepared_data.val.equals(dataset.prepared_data.val)
        assert loaded_dataset.prepared_data.test.equals(dataset.prepared_data.test)


@patch(
    "sotai.pipeline.post_pipeline", return_value=(APIStatus.SUCCESS, "test_pipeline_id")
)
def test_publish(
    post_pipeline,
    test_feature_names: fixture_test_feature_names,
    test_target: fixture_test_target,
):
    """Tests that a pipeline can be published to the API."""
    pipeline = Pipeline(test_feature_names, test_target, TargetType.CLASSIFICATION)
    pipeline_uuid = pipeline.publish()
    post_pipeline.assert_called_once()
    assert pipeline_uuid == "test_pipeline_id"


@patch("sotai.pipeline.Pipeline.upload_model", return_value=APIStatus.SUCCESS)
@patch(
    "sotai.pipeline.post_trained_model_analysis",
    return_value=(
        APIStatus.SUCCESS,
        {"trainedModelMetadataUUID": "test_uuid"},
    ),
)
@patch("sotai.pipeline.post_pipeline_feature_configs", return_value=APIStatus.SUCCESS)
@patch(
    "sotai.pipeline.post_pipeline_config",
    return_value=(APIStatus.SUCCESS, "test_pipeline_config_id"),
)
@patch(
    "sotai.pipeline.post_pipeline", return_value=(APIStatus.SUCCESS, "test_pipeline_id")
)
@patch("sotai.pipeline.get_api_key", return_value="test_api_key")
def test_analysis(
    get_api_key,
    post_pipeline,
    post_pipeline_config,
    post_pipeline_feature_configs,
    post_trained_model_analysis,
    upload_model,
    test_data,
    test_feature_names: fixture_test_feature_names,
    test_target: fixture_test_target,
):
    """Tests that pipeline analysis works as expected."""
    pipeline = Pipeline(test_feature_names, test_target, TargetType.CLASSIFICATION)
    trained_model = pipeline.train(test_data)
    analysis_response = pipeline.analysis(trained_model)

    get_api_key.assert_called()
    upload_model.assert_called_once()
    post_pipeline.assert_called_once()
    post_pipeline_config.assert_called_once_with(
        "test_pipeline_id", trained_model.pipeline_config
    )
    post_pipeline_feature_configs.assert_called_once_with(
        "test_pipeline_config_id", trained_model.pipeline_config.feature_configs
    )
    post_trained_model_analysis.assert_called_once_with(
        "test_pipeline_config_id", trained_model
    )

    assert (
        analysis_response
        == "https://app.sotai.ai/pipelines/test_pipeline_id/trained-models/test_uuid"
    )


@patch(
    "sotai.pipeline.post_trained_model",
    return_value=APIStatus.SUCCESS,
)
@patch("sotai.pipeline.get_api_key", return_value="test_api_key")
def test_upload_model(
    get_api_key,
    post_trained_model,
    test_data: fixture_test_data,
    test_feature_names: fixture_test_feature_names,
    test_target: fixture_test_target,
    tmp_path,
):
    """Tests that a pipeline can be uploaded to the API."""
    pipeline = Pipeline(test_feature_names, test_target, TargetType.CLASSIFICATION)
    trained_model = pipeline.train(test_data)
    trained_model.uuid = "test_uuid"
    upload_model_response = pipeline.upload_model(trained_model, tmp_path)

    get_api_key.assert_called_once()
    post_trained_model.assert_called_once()
    assert trained_model.has_uploaded
    assert upload_model_response == APIStatus.SUCCESS


@patch(
    "sotai.pipeline.post_inference",
    return_value=(APIStatus.SUCCESS, "test_inference_uuid"),
)
@patch("sotai.pipeline.get_api_key", return_value="test_api_key")
def test_run_inference(
    get_api_key,
    post_inference,
    test_data: fixture_test_data,
    test_feature_names: fixture_test_feature_names,
    test_target: fixture_test_target,
):
    """Tests that a pipeline can run inference on a dataset."""
    pipeline = Pipeline(test_feature_names, test_target, TargetType.CLASSIFICATION)
    trained_model = pipeline.train(test_data)
    trained_model.uuid = "test_uuid"

    pipeline.inference("/tmp/data.csv", trained_model.uuid)

    get_api_key.assert_called_once()
    post_inference.assert_called_once_with("/tmp/data.csv", "test_uuid")


@patch("sotai.pipeline.INFERENCE_POLLING_INTERVAL", 1)
@patch("sotai.pipeline.get_inference_results", return_value=APIStatus.SUCCESS)
@patch(
    "sotai.pipeline.get_inference_status",
    side_effect=[
        (APIStatus.SUCCESS, InferenceConfigStatus.INITIALIZING),
        (APIStatus.SUCCESS, InferenceConfigStatus.SUCCESS),
    ],
)
def test_await_inference_results(
    get_inference_status,
    get_inference_results,
    test_data: fixture_test_data,
    test_feature_names: fixture_test_feature_names,
    test_target: fixture_test_target,
):
    """Tests that a pipeline can await inference results."""
    pipeline = Pipeline(test_feature_names, test_target, TargetType.CLASSIFICATION)
    pipeline.train(test_data)

    pipeline.await_inference("test_uuid")
    get_inference_status.assert_called_with("test_uuid")
    get_inference_results.assert_called_once()
