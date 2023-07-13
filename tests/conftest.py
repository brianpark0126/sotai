"""Fixtures to help with testing."""
# pylint: disable=redefined-outer-name
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
from sotai.features import NumericalFeature
from sotai.models import CalibratedLinear
from sotai.types import FeatureAnalysis, FeatureType, LinearConfig, LossType


@pytest.fixture(scope="session")
def fixture_target():
    """Returns a test target."""
    return "target"


@pytest.fixture(scope="session")
def fixture_categories_strs():
    """Returns a list of test string categories."""
    return ["a", "b", "c", "d"]


@pytest.fixture(scope="session")
def fixture_categories_ints():
    """Returns of a list of test integer categories."""
    return [0, 1, 2, 3]


@pytest.fixture(scope="function")
def fixture_data(
    fixture_categories_strs,
    fixture_categories_ints,
    fixture_target,
):
    """Returns a test dataset."""
    return pd.DataFrame(
        {
            "numerical": np.random.rand(100),
            "categorical_strs": np.random.choice(fixture_categories_strs, 100),
            "categorical_ints": np.random.choice(fixture_categories_ints, 100),
            fixture_target: np.random.randint(0, 2, 100),
        }
    )


@pytest.fixture(scope="function")
def fixture_feature_names(fixture_data, fixture_target):
    """Returns a list of test feature names."""
    return fixture_data.columns.drop(fixture_target).to_list()


@pytest.fixture(scope="session")
def fixture_feature_configs(
    fixture_categories_strs,
    fixture_categories_ints,
) -> Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]]:
    """Returns a list of test features."""
    return {
        "numerical": NumericalFeatureConfig(name="numerical"),
        "categorical_strs": CategoricalFeatureConfig(
            name="categorical_strs", categories=fixture_categories_strs
        ),
        "categorical_ints": CategoricalFeatureConfig(
            name="categorical_ints", categories=fixture_categories_ints
        ),
    }


@pytest.fixture(scope="function")
def fixture_pipeline(
    fixture_target,
    fixture_feature_names,
    fixture_categories_strs,
    fixture_categories_ints,
) -> Pipeline:
    """Returns a list of test features."""
    return Pipeline(
        fixture_feature_names,
        fixture_target,
        TargetType.CLASSIFICATION,
        categories={
            "categorical_strs": fixture_categories_strs,
            "categorical_ints": fixture_categories_ints,
        },
    )


@pytest.fixture(scope="function")
def fixture_pipeline_config(
    fixture_target,
    fixture_categories_strs,
    fixture_categories_ints,
) -> PipelineConfig:
    """Returns a pipeline config that can be used for testing."""
    pipeline_config = PipelineConfig(
        id=1,
        shuffle_data=False,
        drop_empty_percentage=80,
        dataset_split={"train": 60, "val": 20, "test": 20},
        target=fixture_target,
        target_type="classification",
        primary_metric="auc",
        feature_configs={
            "numerical": NumericalFeatureConfig(
                name="numerical", monotonicity="increasing"
            ),
            "categorical_strs": CategoricalFeatureConfig(
                name="categorical_strs", categories=fixture_categories_strs
            ),
            "categorical_ints": CategoricalFeatureConfig(
                name="categorical_ints", categories=fixture_categories_ints
            ),
        },
    )
    return pipeline_config


@pytest.fixture(scope="function")
def fixture_trained_model(fixture_pipeline_config) -> TrainedModel:
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
        pipeline_config=fixture_pipeline_config,
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
