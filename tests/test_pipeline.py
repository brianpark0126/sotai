"""Temporary scaffolded test file for pipeline."""
import pandas as pd

from sotai.enums import FeatureType, Metric, TargetType
from sotai.pipeline import Pipeline


def test_pipeline_init():
    """Temporary scaffolded test function."""
    data = pd.DataFrame(
        {"target": [0, 1, 0], "numerical": [0, 1, 2], "categorical": ["a", "b", "c"]}
    )
    target, target_type = "target", TargetType.CLASSIFICATION
    pipeline = Pipeline(data, target, target_type)
    assert pipeline.most_recent_dataset.raw_data.equals(data)
    assert pipeline.target == target
    assert pipeline.target_type == target_type
    assert pipeline.primary_metric == Metric.F1
    assert pipeline.name == f"{target}_{target_type}"
    assert len(pipeline.config.features) == 2
    numerical_feature_config = pipeline.config.features["numerical"]
    assert numerical_feature_config.name == "numerical"
    assert numerical_feature_config.type == FeatureType.NUMERICAL
    assert numerical_feature_config.missing_input_value == -1
    categorical_feature_config = pipeline.config.features["categorical"]
    assert categorical_feature_config.name == "categorical"
    assert categorical_feature_config.type == FeatureType.CATEGORICAL
    assert categorical_feature_config.missing_input_value == "<Missing Value>"
