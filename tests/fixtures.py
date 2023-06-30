from typing import Dict, Union

import numpy as np
import pandas as pd
import pytest

from sotai import (
    CategoricalFeatureConfig,
    NumericalFeatureConfig,
    Pipeline,
    PipelineConfig,
    TargetType,
    TrainedModel,
    TrainingConfig,
    TrainingResults,
)
from sotai.features import CategoricalFeature, NumericalFeature
from sotai.models import CalibratedLinear
from sotai.types import (
    FeatureAnalysis,
    FeatureType,
    LinearConfig,
    LossType,
    Metric,
    TrainingConfig,
    TrainingResults,
)


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


@pytest.fixture(name="test_feature_names")
def fixture_test_feature_names(test_data, test_target):
    """Returns a list of test feature names."""
    return test_data.columns.drop(test_target).to_list()


@pytest.fixture(name="test_feature_configs")
def fixture_test_feature_configs(
    test_categories,
) -> Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]]:
    """Returns a list of test features."""
    return {
        "numerical": NumericalFeatureConfig(name="numerical"),
        "categorical": CategoricalFeatureConfig(
            name="categorical", categories=test_categories
        ),
    }


@pytest.fixture(name="test_pipeline")
def fixture_test_pipeline(test_target, test_feature_names, test_categories) -> Pipeline:

    """Returns a list of test features."""
    return Pipeline(
        test_feature_names,
        test_target,
        TargetType.CLASSIFICATION,
        categories={"categorical": test_categories},
    )


@pytest.fixture(name="test_pipeline_config")
def fixture_test_pipeline_config() -> PipelineConfig:
    """Returns a pipeline config that can be used for testing"""
    pipeline_config = PipelineConfig(
        id=1,
        shuffle_data=False,
        drop_empty_percentage=80,
        dataset_split={"train": 60, "val": 20, "test": 20},
        target="target",
        target_type="classification",
        primary_metric="auc",
        feature_configs={
            "numerical": NumericalFeatureConfig(
                name="numerical", monotonicity="increasing"
            ),
            "categorical": CategoricalFeatureConfig(
                name="categorical", categories=["a", "b", "c", "d"]
            ),
        },
    )
    return pipeline_config


@pytest.fixture(name="test_trained_model")
def fixture_test_trained_model(test_pipeline_config) -> TrainedModel:
    """Returns a trained model that can be used for testing."""

    trained_model = TrainedModel(
        dataset_id=1,
        model=CalibratedLinear(
            features=[
                NumericalFeature(feature_name="test", data=[1, 2, 3, 4, 5, 6, 7, 8])
            ],
        ),
        model_config=LinearConfig(
            use_bias=True,
        ),
        pipeline_config=test_pipeline_config,
        training_config=TrainingConfig(
            loss_type=LossType.MSE,
            epochs=100,
            batch_size=32,
            learning_rate=1e-3,
        ),
        training_results=TrainingResults(
            training_time=1,
            train_loss_by_epoch=[1, 2, 3],
            train_primary_metric_by_epoch=[1, 2, 3],
            val_loss_by_epoch=[1, 2, 3],
            val_primary_metric_by_epoch=[1, 2, 3],
            evaluation_time=1,
            test_loss=1,
            test_primary_metric=1,
            feature_analyses={
                "test": FeatureAnalysis(
                    feature_name="test",
                    feature_type=FeatureType.NUMERICAL,
                    min=1,
                    max=2,
                    mean=3,
                    median=4,
                    std=5,
                    keypoints_inputs_numerical=[1, 2, 3],
                    keypoints_inputs_categorical=None,
                    keypoints_outputs=[1, 2, 3],
                )
            },
            linear_coefficients={"test": 1},
        ),
    )

    return trained_model
