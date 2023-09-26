"""Tests for the shap utils."""
import pandas as pd
import numpy as np

from sotai.utils.shap_utils import (
    calculate_feature_importance,
    calculate_scatter,
    calculate_beeswarm,
)


def test_calculate_feature_importance():
    """Tests the calculate feature importance function."""

    test_shapley_values = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    test_columns = ["a", "b", "c"]
    feature_importance = calculate_feature_importance(test_shapley_values, test_columns)
    print(feature_importance)
    assert feature_importance == [
        {"feature": "c", "value": 4.5},
        {"feature": "b", "value": 3.5},
        {"feature": "a", "value": 2.5},
    ]


def test_calculate_beeswarm():
    """Tests the calculate beeswarm function."""

    test_shapley_values = np.array([[0.1, 0.2], [0.4, 0.5]])

    test_inference_data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    test_target = "c"

    beeswarm_data = calculate_beeswarm(
        test_inference_data, test_shapley_values, test_target
    )
    print(beeswarm_data)
    assert set(beeswarm_data[0]["shaps"]) == set([0.1, 0.4])
    assert set(beeswarm_data[1]["shaps"]) == set([0.2, 0.5])
    assert set(beeswarm_data[0]["pos"]) == set([0.0, 0.0])
    assert set(beeswarm_data[1]["pos"]) == set([1.0, 1.0])
    assert beeswarm_data[0]["cmap"] is None
    assert beeswarm_data[1]["cmap"] is None
    assert beeswarm_data[0]["vmin"] == 1.15
    assert beeswarm_data[1]["vmin"] == 2.15
    assert beeswarm_data[0]["vmax"] == 3.8499999999999996
    assert beeswarm_data[1]["vmax"] == 4.85
    assert set(beeswarm_data[0]["c"]) == set([1.15, 3.8499999999999996])
    assert set(beeswarm_data[1]["c"]) == set([2.15, 4.85])
    assert beeswarm_data[0]["name"] == "a"
    assert beeswarm_data[1]["name"] == "b"


def test_scatter():
    """Tests the calculate scatter function."""

    test_shapley_values = np.array([[0.1, 0.2], [0.4, 0.5]])
    test_inference_data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])

    scatter_data = calculate_scatter(test_inference_data, test_shapley_values)

    print(scatter_data)
    assert scatter_data == [
        {
            "primary_feature_name": "a",
            "colorization_feature_name": "b",
            "x_values": [4, 1],
            "y_values": [0.4, 0.1],
            "colors": [2.0, 5.0],
            "xmin": 0.85,
            "xmax": 4.15,
            "ymin": 2.0,
            "ymax": 5.0,
            "histogram": [1, 0, 0, 0, 1],
            "histogram_bin_edges": [1.0, 1.6, 2.2, 2.8, 3.4, 4.0],
        },
        {
            "primary_feature_name": "b",
            "colorization_feature_name": "a",
            "x_values": [5, 2],
            "y_values": [0.5, 0.2],
            "colors": [1.0, 4.0],
            "xmin": 1.85,
            "xmax": 5.15,
            "ymin": 1.0,
            "ymax": 4.0,
            "histogram": [1, 0, 0, 0, 1],
            "histogram_bin_edges": [2.0, 2.6, 3.2, 3.8, 4.4, 5.0],
        },
    ]
