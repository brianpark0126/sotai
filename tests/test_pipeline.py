"""Temporary scaffolded test file for pipeline."""
import pandas as pd
import pytest

from sotai.enums import FeatureType, Metric, ModelFramework, TargetType
from sotai.pipeline import Pipeline
from sotai.types import LinearOptions, ModelConfig


@pytest.mark.parametrize(
    "target_type,expected_primary_metric",
    [(TargetType.CLASSIFICATION, Metric.AUC), (TargetType.REGRESSION, Metric.MSE)],
)
def test_init(target_type, expected_primary_metric):
    """Tests pipeline initialization for a classification target."""
    features = ["numerical", "categorical"]
    target = "target"
    pipeline = Pipeline(features, target, target_type)
    assert pipeline.target == target
    assert pipeline.target_type == target_type
    assert pipeline.primary_metric == expected_primary_metric
    assert pipeline.name == f"{target}_{target_type}"
    assert pipeline.config
    assert len(pipeline.config.features) == 2
    numerical_config = pipeline.config.features["numerical"]
    assert numerical_config.name == "numerical"
    assert numerical_config.type == FeatureType.NUMERICAL
    categorical_config = pipeline.config.features["categorical"]
    assert categorical_config.name == "categorical"
    # Note: we expect the default config to be numerical if not specified.
    assert categorical_config.type == FeatureType.NUMERICAL


def test_init_with_categories():
    """Tests pipeline initialization with specified categories."""
    features = ["numerical", "categorical"]
    target = "target"
    pipeline = Pipeline(
        features,
        target,
        TargetType.CLASSIFICATION,
        categories={"categorical": ["a", "b", "c"]},
    )
    categorical_config = pipeline.config.features["categorical"]
    assert categorical_config.name == "categorical"
    assert categorical_config.type == FeatureType.CATEGORICAL


def test_prepare():
    """Tests the pipeline prepare function."""
    target = "target"
    data = pd.DataFrame(
        {"numerical": [0, 1, 2], "categorical": ["a", "b", "c"], target: [0, 1, 0]}
    )
    features = data.columns.drop(target).to_list()
    pipeline = Pipeline(features, target, target_type=TargetType.CLASSIFICATION)
    pipeline.config.shuffle_data = False
    pipeline.config.dataset_split.train = 34
    pipeline.config.dataset_split.val = 33
    pipeline.config.dataset_split.test = 33
    dataset_id, pipeline_config_id = pipeline.prepare(data)
    assert pipeline_config_id == 1
    categorical_config = pipeline.configs[pipeline_config_id].features["categorical"]
    assert categorical_config.name == "categorical"
    assert categorical_config.type == FeatureType.CATEGORICAL
    assert categorical_config.categories == ["a", "b", "c"]
    assert dataset_id == 1
    assert pipeline.datasets[dataset_id].pipeline_config_id == pipeline_config_id
    assert pipeline.datasets[dataset_id].prepared_data.train.equals(data.iloc[:1])
    assert pipeline.datasets[dataset_id].prepared_data.val.equals(data.iloc[1:2])
    assert pipeline.datasets[dataset_id].prepared_data.test.equals(data.iloc[2:3])


@pytest.mark.parametrize(
    "model_framework",
    [(ModelFramework.TENSORFLOW), (ModelFramework.PYTORCH)],
)
def test_train_calibrated_linear_classification_model(model_framework):
    """Tests pipeline training for calibrated linear classficiation model."""
    target = "target"
    data = pd.DataFrame(
        {
            target: [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
            "numerical": [0, 1, 2, 1, 6, 4, 3, 5, 6, 3],
            "categorical": ["a", "b", "c", "c", "b", "c", "a", "a", "b", "c"],
        }
    )
    features = data.columns.drop(target).to_list()
    pipeline = Pipeline(features, target, target_type=TargetType.CLASSIFICATION)
    pipeline.config.shuffle_data = False
    pipeline.config.dataset_split.train = 80
    pipeline.config.dataset_split.val = 10
    pipeline.config.dataset_split.test = 10
    dataset_id, pipeline_config_id = pipeline.prepare(data)
    trained_model_id, trained_model = pipeline.train(
        dataset_id,
        pipeline_config_id,
        model_config=ModelConfig(framework=model_framework, options=LinearOptions()),
    )
    assert trained_model_id == 1
    assert trained_model


@pytest.mark.parametrize(
    "model_framework",
    [(ModelFramework.TENSORFLOW), (ModelFramework.PYTORCH)],
)
def test_train_calibrated_linear_regression_model(model_framework):
    """Tests pipeline training for calibrated linear regression model."""
    target = "target"
    data = pd.DataFrame(
        {
            target: [12, 5.6, 4, 23.5, 3.5, 2.7, 7.3, 34, 45.2, 1.0],
            "numerical": [0, 1, 2, 1, 6, 4, 3, 5, 6, 3],
            "categorical": ["a", "b", "c", "c", "b", "c", "a", "a", "b", "c"],
        }
    )
    features = data.columns.drop(target).to_list()
    pipeline = Pipeline(features, target, target_type=TargetType.REGRESSION)
    pipeline.config.shuffle_data = False
    pipeline.config.dataset_split.train = 80
    pipeline.config.dataset_split.val = 10
    pipeline.config.dataset_split.test = 10
    dataset_id, pipeline_config_id = pipeline.prepare(data)
    trained_model_id, trained_model = pipeline.train(
        dataset_id,
        pipeline_config_id,
        model_config=ModelConfig(framework=model_framework, options=LinearOptions()),
    )
    assert trained_model_id == 1
    assert trained_model
