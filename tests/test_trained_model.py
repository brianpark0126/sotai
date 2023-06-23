"""Tests for TrainedModel."""
import warnings
from typing import Dict, Union

import numpy as np
import pandas as pd
import pytest
import pytorch_calibrated as ptcm

from sotai import (
    CategoricalFeature,
    DatasetSplit,
    LinearOptions,
    LossType,
    Metric,
    ModelConfig,
    ModelFramework,
    NumericalFeature,
    PipelineConfig,
    TargetType,
    TrainedModel,
    TrainingConfig,
    TrainingResults,
)
from sotai.modeling_utils import (
    create_ptcm_feature_configs,
    create_ptcm_model,
    create_tfl_feature_configs,
    create_tfl_model,
    prepare_tfl_data,
)


@pytest.fixture(name="categories")
def fixture_categories():
    """Returns a list of categories."""
    return ["a", "b", "c", "d"]


@pytest.fixture(name="data")
def fixture_data(categories):
    """Returns a test dataset."""
    return pd.DataFrame(
        {
            "numerical": np.random.rand(100),
            "categorical": np.random.choice(categories, 100),
        }
    )


@pytest.fixture(name="features")
def fixture_features(categories):
    """Returns a list of features."""
    return {
        "numerical": NumericalFeature(name="numerical"),
        "categorical": CategoricalFeature(name="categorical", categories=categories),
    }


def _construct_trained_model(
    model_framework: ModelFramework,
    target_type: TargetType,
    data: pd.DataFrame,
    features: Dict[str, Union[CategoricalFeature, NumericalFeature]],
):
    """Returns a `TrainedModel` instance."""
    model_config = ModelConfig(
        framework=model_framework, options=LinearOptions(output_calibration=False)
    )
    training_config = TrainingConfig(loss_type=LossType.MSE)
    primary_metric = Metric.MSE

    if model_framework == ModelFramework.TENSORFLOW:
        _, _, x_dict = prepare_tfl_data(data, features, None)
        tfl_feature_configs = create_tfl_feature_configs(features, x_dict)
        model = create_tfl_model(
            tfl_feature_configs,
            model_config,
            training_config,
            primary_metric,
            np.array([]),
            True,
        )
    else:
        ptcm_feature_configs = create_ptcm_feature_configs(
            features, ptcm.data.CSVData(data)
        )
        model = create_ptcm_model(ptcm_feature_configs, model_config)

    return TrainedModel(
        dataset_id=0,
        pipeline_config=PipelineConfig(
            id=0,
            target="target",
            target_type=target_type,
            primary_metric=primary_metric,
            features=features,
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


@pytest.mark.parametrize(
    "model_framework", [(ModelFramework.TENSORFLOW), (ModelFramework.PYTORCH)]
)
def test_trained_classification_model_predict(data, features, model_framework):
    """Tests the predict function on a trained model."""
    trained_model = _construct_trained_model(
        model_framework, TargetType.CLASSIFICATION, data, features
    )
    predictions, probabilities = trained_model.predict(data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(data)
    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == len(data)


@pytest.mark.parametrize(
    "model_framework", [(ModelFramework.TENSORFLOW), (ModelFramework.PYTORCH)]
)
def test_trained_regression_model_predict(data, features, model_framework):
    """Tests the predict function on a trained model."""
    trained_model = _construct_trained_model(
        model_framework, TargetType.REGRESSION, data, features
    )
    predictions = trained_model.predict(data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(data)
