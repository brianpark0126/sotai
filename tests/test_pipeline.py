"""Temporary scaffolded test file for pipeline."""
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from sotai.enums import FeatureType, Metric, ModelFramework, TargetType
from sotai.pipeline import Pipeline
from sotai.types import LinearOptions, ModelConfig


def _create_pipeline_and_train_model(
    data: pd.DataFrame,
    features: List[str],
    target: str,
    target_type: TargetType,
    model_framework: ModelFramework,
) -> Tuple[Pipeline, int]:
    """Returns a pipeline and trained model id."""
    pipeline = Pipeline(features, target, target_type)
    pipeline.config.shuffle_data = False
    pipeline.config.dataset_split.train = 60
    pipeline.config.dataset_split.val = 20
    pipeline.config.dataset_split.test = 20
    dataset_id, pipeline_config_id = pipeline.prepare(data)
    trained_model_id, _ = pipeline.train(
        dataset_id,
        pipeline_config_id,
        model_config=ModelConfig(framework=model_framework, options=LinearOptions()),
    )
    return pipeline, trained_model_id


@pytest.fixture(name="test_target")
def fixture_test_target():
    """Returns a test target."""
    return "target"


@pytest.fixture(name="test_categories")
def fixture_test_categories():
    """Returns a list of test categories."""
    return ["a", "b", "c", "d"]


@pytest.fixture(name="test_data")
def fixture_test_data(test_categories, test_target):
    """Returns a test dataset."""
    return pd.DataFrame(
        {
            "numerical": np.random.rand(100),
            "categorical": np.random.choice(test_categories, 100),
            test_target: np.random.randint(0, 2, 100),
        }
    )


@pytest.fixture(name="test_features")
def fixture_test_features(test_data, test_target):
    """Returns a list of test features."""
    return test_data.columns.drop(test_target).to_list()


@pytest.mark.parametrize(
    "target_type,expected_primary_metric",
    [(TargetType.CLASSIFICATION, Metric.AUC), (TargetType.REGRESSION, Metric.MSE)],
)
def test_init(test_features, test_target, target_type, expected_primary_metric):
    """Tests pipeline initialization for a classification target."""
    pipeline = Pipeline(test_features, test_target, target_type)
    assert pipeline.target == test_target
    assert pipeline.target_type == target_type
    assert pipeline.primary_metric == expected_primary_metric
    assert pipeline.name == f"{test_target}_{target_type}"
    assert pipeline.config
    assert len(pipeline.config.features) == 2
    numerical_config = pipeline.config.features["numerical"]
    assert numerical_config.name == "numerical"
    assert numerical_config.type == FeatureType.NUMERICAL
    categorical_config = pipeline.config.features["categorical"]
    assert categorical_config.name == "categorical"
    # Note: we expect the default config to be numerical if not specified.
    assert categorical_config.type == FeatureType.NUMERICAL


def test_init_with_categories(test_features, test_target, test_categories):
    """Tests pipeline initialization with specified categories."""
    pipeline = Pipeline(
        test_features,
        test_target,
        TargetType.CLASSIFICATION,
        categories={"categorical": test_categories},
    )
    categorical_config = pipeline.config.features["categorical"]
    assert categorical_config.name == "categorical"
    assert categorical_config.type == FeatureType.CATEGORICAL


def test_prepare(test_data, test_features, test_target, test_categories):
    """Tests the pipeline prepare function."""
    pipeline = Pipeline(
        test_features, test_target, target_type=TargetType.CLASSIFICATION
    )
    # We set shuffle to false to ensure the data is split in the same way.
    pipeline.config.shuffle_data = False
    pipeline.config.dataset_split.train = 80
    pipeline.config.dataset_split.val = 10
    pipeline.config.dataset_split.test = 10
    dataset_id, pipeline_config_id = pipeline.prepare(test_data)
    assert pipeline_config_id == 1
    categorical_config = pipeline.configs[pipeline_config_id].features["categorical"]
    assert categorical_config.name == "categorical"
    assert categorical_config.type == FeatureType.CATEGORICAL
    assert categorical_config.categories == test_categories
    assert dataset_id == 1
    assert pipeline.datasets[dataset_id].pipeline_config_id == pipeline_config_id
    num_examples = len(test_data)
    num_training_examples = int(
        num_examples * pipeline.config.dataset_split.train / 100
    )
    num_val_examples = int(num_examples * pipeline.config.dataset_split.val / 100)
    assert pipeline.datasets[dataset_id].prepared_data.train.equals(
        test_data.iloc[:num_training_examples]
    )
    assert pipeline.datasets[dataset_id].prepared_data.val.equals(
        test_data.iloc[num_training_examples : num_training_examples + num_val_examples]
    )
    assert pipeline.datasets[dataset_id].prepared_data.test.equals(
        test_data.iloc[num_training_examples + num_val_examples :]
    )


@pytest.mark.parametrize(
    "model_framework",
    [(ModelFramework.TENSORFLOW), (ModelFramework.PYTORCH)],
)
def test_train_calibrated_linear_classification_model(
    test_data, test_features, test_target, model_framework
):
    """Tests pipeline training for calibrated linear classficiation model."""
    pipeline, trained_model_id = _create_pipeline_and_train_model(
        test_data,
        test_features,
        test_target,
        TargetType.CLASSIFICATION,
        model_framework,
    )
    assert trained_model_id == 1
    assert pipeline.models[trained_model_id]


@pytest.mark.parametrize(
    "model_framework",
    [(ModelFramework.TENSORFLOW), (ModelFramework.PYTORCH)],
)
def test_train_calibrated_linear_regression_model(
    test_data, test_features, test_target, model_framework
):
    """Tests pipeline training for calibrated linear regression model."""
    pipeline, trained_model_id = _create_pipeline_and_train_model(
        test_data,
        test_features,
        test_target,
        TargetType.REGRESSION,
        model_framework,
    )
    assert trained_model_id == 1
    assert pipeline.models[trained_model_id]


@pytest.mark.parametrize(
    "model_framework",
    [(ModelFramework.TENSORFLOW), (ModelFramework.PYTORCH)],
)
def test_pipeline_classification_predict(
    test_data, test_features, test_target, model_framework
):
    """Tests the pipeline predict function on a trained model."""
    pipeline, trained_model_id = _create_pipeline_and_train_model(
        test_data,
        test_features,
        test_target,
        TargetType.CLASSIFICATION,
        model_framework,
    )
    predictions, probabilities = pipeline.predict(test_data, trained_model_id)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test_data)
    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == len(test_data)


@pytest.mark.parametrize(
    "model_framework",
    [(ModelFramework.TENSORFLOW), (ModelFramework.PYTORCH)],
)
def test_pipeline_regression_predict(
    test_data, test_features, test_target, model_framework
):
    """Tests the pipeline predict function on a trained model."""
    pipeline, trained_model_id = _create_pipeline_and_train_model(
        test_data,
        test_features,
        test_target,
        TargetType.REGRESSION,
        model_framework,
    )
    predictions = pipeline.predict(test_data, trained_model_id)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test_data)
