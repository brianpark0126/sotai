"""Temporary scaffolded test file for pipeline utils."""
from datetime import datetime

import pandas as pd

from sotai.enums import FeatureType
from sotai.utils import default_feature_configs


def test_default_feature_configs():
    """Tests the generation of default feature configs."""
    target = "target"
    data = pd.DataFrame(
        {
            target: [0, 1, 0],
            "numerical": [0, 1, 2],
            "categorical": ["a", "b", "c"],
            "unknown": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 2),
                datetime(2020, 1, 3),
            ],
        }
    )
    categories = ["categorical"]
    feature_configs = default_feature_configs(data, target, categories)
    assert len(feature_configs) == 2
    numerical_feature_config = feature_configs["numerical"]
    assert numerical_feature_config.name == "numerical"
    assert numerical_feature_config.type == FeatureType.NUMERICAL
    assert numerical_feature_config.missing_input_value == -1
    categorical_feature_config = feature_configs["categorical"]
    assert categorical_feature_config.name == "categorical"
    assert categorical_feature_config.type == FeatureType.CATEGORICAL
    assert categorical_feature_config.missing_input_value == "<Missing Value>"
