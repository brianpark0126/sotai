"""Temporary scaffolded test file for pipeline utils."""
from datetime import datetime

import pandas as pd

from sotai.enums import FeatureType
from sotai.utils import determine_feature_types


def test_determine_feature_types():
    """Tests the determination of feature types from data."""
    data = pd.DataFrame(
        {
            "numerical": [0, 1, 2],
            "categorical": ["a", "b", "c"],
            "unknown": [datetime.now(), datetime.now(), datetime.now()],
        }
    )
    feature_types = determine_feature_types(data)
    assert feature_types["numerical"] == FeatureType.NUMERICAL
    assert feature_types["categorical"] == FeatureType.CATEGORICAL
    assert feature_types["unknown"] == FeatureType.UNKNOWN
