"""Temporary scaffolded test file for pipeline."""
import pandas as pd

from sotai.enums import Metric, TargetType
from sotai.pipeline import Pipeline


def test_pipeline_init_classification():
    """Tests pipeline initialization for a classification target."""
    target = "target"
    data = pd.DataFrame(
        {target: [0, 1, 0], "numerical": [0, 1, 2], "categorical": ["a", "b", "c"]}
    )
    pipeline = Pipeline(data, target)
    assert pipeline.target == target
    assert pipeline.target_type == TargetType.CLASSIFICATION
    assert pipeline.primary_metric == Metric.F1
    assert pipeline.name == f"{target}_{TargetType.CLASSIFICATION}"
    assert pipeline.config


def test_pipeline_init_regression():
    """Tests pipeline initialization for a regression target."""
    target = "target"
    data = pd.DataFrame(
        {
            target: [0.0, 1.0, 0.0],
            "numerical": [0, 1, 2],
            "categorical": ["a", "b", "c"],
        }
    )
    pipeline = Pipeline(data, target)
    assert pipeline.target == target
    assert pipeline.target_type == TargetType.REGRESSION
    assert pipeline.primary_metric == Metric.MSE
    assert pipeline.name == f"{target}_{TargetType.REGRESSION}"
    assert pipeline.config


def test_pipeline_prepare():
    """Tests the pipeline prepare function."""
    target = "target"
    data = pd.DataFrame(
        {target: [0, 1, 0], "numerical": [0, 1, 2], "categorical": ["a", "b", "c"]}
    )
    pipeline = Pipeline(data, target, target_type=TargetType.CLASSIFICATION)
    pipeline.config.shuffle_data = False
    pipeline.config.dataset_split.train = 34
    pipeline.config.dataset_split.val = 33
    pipeline.config.dataset_split.test = 33
    dataset_id, pipeline_config_id = pipeline.prepare(data)
    assert dataset_id == 1
    assert pipeline_config_id == 1
    assert pipeline.datasets[dataset_id].pipeline_config_id == pipeline_config_id
    assert set(pipeline.datasets[dataset_id].columns) == set(data.columns)
    assert pipeline.datasets[dataset_id].prepared_data.train.equals(data.iloc[:1])
    assert pipeline.datasets[dataset_id].prepared_data.val.equals(data.iloc[1:2])
    assert pipeline.datasets[dataset_id].prepared_data.test.equals(data.iloc[2:3])
