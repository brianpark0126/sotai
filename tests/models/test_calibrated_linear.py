"""Tests for calibrated linear model."""
import numpy as np
import pytest
import torch

from sotai import Monotonicity
from sotai.features import CategoricalFeature, NumericalFeature
from sotai.models import CalibratedLinear
from sotai.models.model_utils import calibrate_and_stack

from ..utils import train_calibrated_module
from unittest.mock import patch, Mock


@pytest.mark.parametrize(
    "features,output_min,output_max,output_calibration_num_keypoints,"
    "expected_linear_monotonicities,expected_output_calibrator_monotonicity",
    [
        (
            [
                NumericalFeature(
                    feature_name="numerical_feature",
                    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    num_keypoints=5,
                    monotonicity=Monotonicity.NONE,
                ),
                CategoricalFeature(
                    feature_name="categorical_feature",
                    categories=["a", "b", "c"],
                    monotonicity_pairs=[("a", "b")],
                ),
            ],
            None,
            None,
            None,
            [
                Monotonicity.NONE,
                Monotonicity.INCREASING,
            ],
            Monotonicity.INCREASING,
        ),
        (
            [
                NumericalFeature(
                    feature_name="numerical_feature",
                    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    num_keypoints=5,
                    monotonicity=Monotonicity.NONE,
                ),
                CategoricalFeature(
                    feature_name="categorical_feature",
                    categories=["a", "b", "c"],
                    monotonicity_pairs=None,
                ),
            ],
            -1.0,
            1.0,
            10,
            [
                Monotonicity.INCREASING,
                Monotonicity.INCREASING,
            ],
            Monotonicity.NONE,
        ),
        (
            [
                NumericalFeature(
                    feature_name="numerical_feature",
                    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    num_keypoints=5,
                    monotonicity=Monotonicity.DECREASING,
                ),
                CategoricalFeature(
                    feature_name="categorical_feature",
                    categories=["a", "b", "c"],
                    monotonicity_pairs=None,
                ),
            ],
            0.0,
            None,
            None,
            [
                Monotonicity.INCREASING,
                Monotonicity.INCREASING,
            ],
            Monotonicity.INCREASING,
        ),
    ],
)
def test_initialization(
    features,
    output_min,
    output_max,
    output_calibration_num_keypoints,
    expected_linear_monotonicities,
    expected_output_calibrator_monotonicity,
):
    """Tests that `CalibratedLinear` initialization works."""
    calibrated_linear = CalibratedLinear(
        features=features,
        output_min=output_min,
        output_max=output_max,
        output_calibration_num_keypoints=output_calibration_num_keypoints,
    )
    assert calibrated_linear.features == features
    assert calibrated_linear.output_min == output_min
    assert calibrated_linear.output_max == output_max
    assert (
        calibrated_linear.output_calibration_num_keypoints
        == output_calibration_num_keypoints
    )
    assert len(calibrated_linear.calibrators) == len(features)
    for calibrator in calibrated_linear.calibrators.values():
        assert calibrator.output_min == output_min
        assert calibrator.output_max == output_max
    assert calibrated_linear.linear.monotonicities == expected_linear_monotonicities
    if (
        output_min is not None
        or output_max is not None
        or output_calibration_num_keypoints
    ):
        assert not calibrated_linear.linear.use_bias
        assert calibrated_linear.linear.weighted_average
    else:
        assert calibrated_linear.linear.use_bias
        assert not calibrated_linear.linear.weighted_average
    if not output_calibration_num_keypoints:
        assert calibrated_linear.output_calibrator is None
    else:
        assert calibrated_linear.output_calibrator.output_min == output_min
        assert calibrated_linear.output_calibrator.output_max == output_max
        assert (
            calibrated_linear.output_calibrator.monotonicity
            == expected_output_calibrator_monotonicity
        )


# TODO: add more parameterized tests
@pytest.mark.parametrize(
    "output_min,output_max,calibrator_kernel_datas,linear_kernel_data,"
    "output_calibrator_kernel_data,inputs,expected_outputs",
    [
        (
            None,
            None,
            [
                torch.tensor([[0.0], [1.0], [1.0], [1.0]]).double(),
                torch.tensor([[1.0], [2.0], [3.0]]).double(),
            ],
            torch.tensor([[1.0], [2.0]]).double(),
            None,
            torch.tensor([[1.0, 0], [2.0, 1], [3.0, 2], [4.0, 1]]).double(),
            torch.tensor([[2.0], [5.0], [8.0], [7.0]]).double(),
        ),
        (
            0.0,
            1.0,
            [
                torch.tensor([[1.0], [-0.5], [-0.5], [1.0]]).double(),
                torch.tensor([[1.0], [0.5], [0.0]]).double(),
            ],
            torch.tensor([[0.3], [0.7]]).double(),
            torch.tensor([[0.0], [1.0], [-1.0], [1.0]]).double(),
            torch.tensor([[1.0, 0], [2.0, 1], [3.0, 2], [4.0, 1]]).double(),
            torch.tensor([[1.0], [0.5], [0.0], [0.05]]).double(),
        ),
    ],
)
def test_forward(
    output_min,
    output_max,
    calibrator_kernel_datas,
    linear_kernel_data,
    output_calibrator_kernel_data,
    inputs,
    expected_outputs,
):
    """Tests that forward returns expected result."""
    calibrated_linear = CalibratedLinear(
        features=[
            NumericalFeature(
                feature_name="numerical_feature",
                data=np.array([1.0, 2.0, 3.0, 4.0]),
                num_keypoints=4,
                monotonicity=Monotonicity.NONE,
            ),
            CategoricalFeature(
                feature_name="categorical_feature",
                categories=["a", "b", "c"],
                monotonicity_pairs=None,
            ),
        ],
        output_min=output_min,
        output_max=output_max,
        output_calibration_num_keypoints=output_calibrator_kernel_data.size()[0]
        if output_calibrator_kernel_data is not None
        else None,
    )
    for i, calibrator in enumerate(calibrated_linear.calibrators.values()):
        calibrator.kernel.data = calibrator_kernel_datas[i]
    calibrated_linear.linear.kernel.data = linear_kernel_data
    if output_calibrator_kernel_data is not None:
        calibrated_linear.output_calibrator.kernel.data = output_calibrator_kernel_data
    outputs = calibrated_linear(inputs)
    assert torch.allclose(outputs, expected_outputs)


def test_linear_assert_constraints():
    """Tests `assert_constraints()` method calls internal assert_constraints."""
    calibrated_linear = CalibratedLinear(
        features=[
            NumericalFeature(
                feature_name="numerical_feature",
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                num_keypoints=5,
                monotonicity=Monotonicity.NONE,
            ),
            CategoricalFeature(
                feature_name="categorical_feature",
                categories=["a", "b", "c"],
                monotonicity_pairs=[("a", "b")],
            ),
        ]
    )

    with patch.object(
        calibrated_linear.linear, "assert_constraints", Mock()
    ) as mock_linear_assert_constraints:
        mock_asserts = []
        for calibrator in calibrated_linear.calibrators.values():
            mock_assert = Mock()
            calibrator.assert_constraints = mock_assert
            mock_asserts.append(mock_assert)

        calibrated_linear.assert_constraints()

        mock_linear_assert_constraints.assert_called_once()
        for mock_assert in mock_asserts:
            mock_assert.assert_called_once()


def test_linear_constrain():
    """Tests `constrain()` method calls internal constrain functions."""
    calibrated_linear = CalibratedLinear(
        features=[
            NumericalFeature(
                feature_name="numerical_feature",
                data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                num_keypoints=5,
                monotonicity=Monotonicity.NONE,
            ),
            CategoricalFeature(
                feature_name="categorical_feature",
                categories=["a", "b", "c"],
                monotonicity_pairs=[("a", "b")],
            ),
        ],
        output_calibration_num_keypoints=2,
    )

    with patch.object(
        calibrated_linear.linear, "constrain", Mock()
    ) as mock_lattice_constrain:
        with patch.object(
            calibrated_linear.output_calibrator, "constrain", Mock()
        ) as mock_output_calibrator_constrain:
            mock_constrains = []
            for calibrator in calibrated_linear.calibrators.values():
                mock_calibrator_constrain = Mock()
                calibrator.constrain = mock_calibrator_constrain
                mock_constrains.append(mock_calibrator_constrain)

            calibrated_linear.constrain()

            mock_lattice_constrain.assert_called_once()
            mock_output_calibrator_constrain.assert_called_once()
            for mock_constrain in mock_constrains:
                mock_constrain.assert_called_once()


def test_training():  # pylint: disable=too-many-locals
    """Tests `CalibratedLinear` training on data from f(x) = 0.7|x_1| + 0.3x_2."""
    num_examples, num_categories = 3000, 3
    output_min, output_max = 0.0, num_categories - 1
    x_1_numpy = np.random.uniform(-output_max, output_max, size=num_examples)
    x_1 = torch.from_numpy(x_1_numpy)[:, None]
    num_examples_per_category = num_examples // num_categories
    x2_numpy = np.concatenate(
        [[c] * num_examples_per_category for c in range(num_categories)]
    )
    x_2 = torch.from_numpy(x2_numpy)[:, None]
    training_examples = torch.column_stack((x_1, x_2))
    linear_coefficients = torch.tensor([0.7, 0.3]).double()
    training_labels = torch.sum(
        torch.column_stack((torch.absolute(x_1), x_2)) * linear_coefficients,
        dim=1,
        keepdim=True,
    )
    randperm = torch.randperm(training_examples.size()[0])
    training_examples = training_examples[randperm]
    training_labels = training_labels[randperm]

    calibrated_linear = CalibratedLinear(
        features=[
            NumericalFeature(
                "x1",
                x_1_numpy,
                num_keypoints=4,
            ),
            CategoricalFeature("x2", [0, 1, 2], monotonicity_pairs=[(0, 1), (1, 2)]),
        ],
        output_min=output_min,
        output_max=output_max,
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(calibrated_linear.parameters(recurse=True), lr=1e-1)

    with torch.no_grad():
        initial_predictions = calibrated_linear(training_examples)
        initial_loss = loss_fn(initial_predictions, training_labels)

    train_calibrated_module(
        calibrated_linear,
        training_examples,
        training_labels,
        loss_fn,
        optimizer,
        500,
        num_examples // 10,
    )

    with torch.no_grad():
        trained_predictions = calibrated_linear(training_examples)
        trained_loss = loss_fn(trained_predictions, training_labels)

    assert trained_loss < initial_loss
    assert trained_loss < 0.02
