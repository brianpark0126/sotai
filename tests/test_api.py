from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from requests import Response

from sotai import (CategoricalFeatureConfig, NumericalFeatureConfig, Pipeline,
                   TargetType)
from sotai.api import (post_pipeline, post_pipeline_config,
                       post_trained_model_analysis)
from sotai.constants import SOTAI_API_ENDPOINT

from .utils import construct_trained_model


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
def fixture_test_feature_configs(test_categories):
    """Returns a list of test features."""
    return {
        "numerical": NumericalFeatureConfig(name="numerical"),
        "categorical": CategoricalFeatureConfig(
            name="categorical", categories=test_categories
        ),
    }


class MockResponse:
    def __init__(self, json_data):
        self.json_data = json_data

    def json(self):
        return self.json_data


@patch("requests.post", return_value=MockResponse({"uuid": "test_uuid"}))
@patch("sotai.api.get_api_key", return_value="test_api_key")
def test_post_pipeline(
    mock_get_api_key, mock_post, test_feature_names, test_target, test_categories
):
    pipeline = Pipeline(
        test_feature_names,
        test_target,
        TargetType.CLASSIFICATION,
        categories={"categorical": test_categories},
    )

    pipeline_uuid = post_pipeline(pipeline)

    mock_post.assert_called_with(
        f"{SOTAI_API_ENDPOINT}/api/v1/public/pipeline",
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
def test_post_pipeline_config():
    pass


def test_post_trained_model(test_data, test_feature_configs):
    trained_model = construct_trained_model(
        TargetType.CLASSIFICATION, test_data, test_feature_configs
    )
    pass
