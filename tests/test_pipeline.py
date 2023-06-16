"""Temporary scaffolded test file for pipeline."""
import pandas as pd

from sotai.enums import FeatureType, Metric, TargetType
from sotai.pipeline import Pipeline


def test_pipeline_init():
    """Temporary scaffolded test function."""
    target, target_type = "target", TargetType.CLASSIFICATION
    data = pd.DataFrame(
        {target: [0, 1, 0], "numerical": [0, 1, 2], "categorical": ["a", "b", "c"]}
    )
    pipeline = Pipeline(data, target, target_type)
    assert pipeline.most_recent_dataset.raw_data.equals(data)
    assert pipeline.target == target
    assert pipeline.target_type == target_type
    assert pipeline.primary_metric == Metric.F1
    assert pipeline.name == f"{target}_{target_type}"
    assert pipeline.config
