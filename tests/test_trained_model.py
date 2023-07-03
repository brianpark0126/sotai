"""Tests for Trained Model."""

import numpy as np

from sotai import TargetType, TrainedModel
from sotai.models import CalibratedLinear


from .fixtures import (  # pylint: disable=unused-import
    fixture_test_categories,
    fixture_test_data,
    fixture_test_feature_configs,
    fixture_test_feature_names,
    fixture_test_target,
)
from .utils import construct_trained_model


def test_trained_classification_model_predict(
    test_data: fixture_test_data, test_feature_configs: fixture_test_feature_configs
):
    """Tests the predict function on a trained model."""
    trained_model = construct_trained_model(
        TargetType.CLASSIFICATION, test_data, test_feature_configs
    )
    predictions, probabilities = trained_model.predict(test_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test_data)
    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == len(test_data)


def test_trained_regression_model_predict(
    test_data: fixture_test_data, test_feature_configs: fixture_test_feature_configs
):
    """Tests the predict function on a trained model."""
    trained_model = construct_trained_model(
        TargetType.REGRESSION, test_data, test_feature_configs
    )
    predictions = trained_model.predict(test_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test_data)


def test_trained_model_save_load(
    test_data: fixture_test_data,
    test_feature_configs: fixture_test_feature_configs,
    tmp_path,
):
    """Tests that a `TrainedModel` can be successfully saved and then loaded."""
    trained_model = construct_trained_model(
        TargetType.CLASSIFICATION, test_data, test_feature_configs
    )
    trained_model.save(tmp_path)
    loaded_trained_model = TrainedModel.load(tmp_path)
    assert isinstance(loaded_trained_model, TrainedModel)
    assert loaded_trained_model.dict(exclude={"model"}) == trained_model.dict(
        exclude={"model"}
    )
    assert isinstance(loaded_trained_model.model, CalibratedLinear)
