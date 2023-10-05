"""Tests for calibrated lattice model."""
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
import pytest
import torch

from sotai import Monotonicity, CSVData, Interpolation, LatticeInit
from sotai.features import CategoricalFeature, NumericalFeature
from sotai.models import CalibratedLattice

from ..utils import train_calibrated_module, train_calibrated_module_tqdm


def test_init_required_args():
    """Tests `CalibratedLattice` initialization with only required arguments."""
    calibrated_lattice = CalibratedLattice(
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
    assert calibrated_lattice.clip_inputs
    assert calibrated_lattice.output_min is None
    assert calibrated_lattice.output_max is None
    assert calibrated_lattice.kernel_init == LatticeInit.LINEAR
    assert calibrated_lattice.interpolation == Interpolation.HYPERCUBE
    assert calibrated_lattice.lattice.lattice_sizes == [2, 2]
    assert calibrated_lattice.output_calibration_num_keypoints is None
    assert calibrated_lattice.output_calibrator is None
    for calibrator in calibrated_lattice.calibrators.values():
        assert calibrator.output_min == 0.0
        assert calibrator.output_max == 1.0


@pytest.mark.parametrize(
    "features, output_min, output_max, interpolation, output_num_keypoints,"
    "expected_monotonicity, expected_lattice_sizes, expected_output_monotonicity",
    [
        (
            [
                NumericalFeature(
                    feature_name="numerical_feature",
                    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    num_keypoints=5,
                    monotonicity=Monotonicity.DECREASING,
                    output_keypoints=3,
                ),
                CategoricalFeature(
                    feature_name="categorical_feature",
                    categories=["a", "b", "c"],
                    monotonicity_pairs=[("a", "b")],
                    output_keypoints=2,
                ),
            ],
            0.5,
            2.0,
            Interpolation.SIMPLEX,
            4,
            [Monotonicity.INCREASING, Monotonicity.INCREASING],
            [3, 2],
            Monotonicity.INCREASING,
        ),
        (
            [
                NumericalFeature(
                    feature_name="numerical_feature",
                    data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                    num_keypoints=5,
                    output_keypoints=4,
                ),
                CategoricalFeature(
                    feature_name="categorical_feature",
                    categories=["a", "b", "c"],
                    output_keypoints=4,
                ),
            ],
            -0.5,
            8.0,
            Interpolation.HYPERCUBE,
            5,
            [Monotonicity.NONE, Monotonicity.NONE],
            [4, 4],
            Monotonicity.NONE,
        ),
    ],
)
def test_init_full_args(
    features,
    output_min,
    output_max,
    interpolation,
    output_num_keypoints,
    expected_monotonicity,
    expected_lattice_sizes,
    expected_output_monotonicity,
):
    """Tests `CalibratedLattice` initialization with all arguments."""
    calibrated_lattice = CalibratedLattice(
        features=features,
        output_min=output_min,
        output_max=output_max,
        interpolation=interpolation,
        output_calibration_num_keypoints=output_num_keypoints,
    )
    assert calibrated_lattice.clip_inputs
    assert calibrated_lattice.output_min == output_min
    assert calibrated_lattice.output_max == output_max
    assert calibrated_lattice.interpolation == interpolation
    assert calibrated_lattice.output_calibration_num_keypoints == output_num_keypoints
    assert calibrated_lattice.output_calibrator.output_min == output_min
    assert calibrated_lattice.output_calibrator.output_max == output_max
    assert (
        calibrated_lattice.output_calibrator.monotonicity
        == expected_output_monotonicity
    )
    assert calibrated_lattice.monotonicities == expected_monotonicity
    assert calibrated_lattice.lattice.lattice_sizes == expected_lattice_sizes
    for calibrator, lattice_dim in zip(
        calibrated_lattice.calibrators.values(), expected_lattice_sizes
    ):
        assert calibrator.output_min == 0.0
        assert calibrator.output_max == lattice_dim - 1


def test_forward():
    """Tests all parts of calibrated lattice forward pass are called."""
    calibrated_lattice = CalibratedLattice(
        features=[
            NumericalFeature(
                feature_name="n",
                data=np.array([1.0, 2.0]),
            ),
            CategoricalFeature(
                feature_name="c",
                categories=["a", "b", "c"],
            ),
        ],
        output_calibration_num_keypoints=10,
    )

    with patch(
        "sotai.models.calibrated_lattice.calibrate_and_stack",
        return_value=torch.tensor([[0.0]]),
    ) as mock_calibrate_and_stack, patch.object(
        calibrated_lattice.lattice, "forward", return_value=torch.tensor([[0.0]])
    ) as mock_lattice, patch.object(
        calibrated_lattice.output_calibrator,
        "forward",
        return_value=torch.tensor([[0.0]]),
    ) as mock_output_calibrator:
        calibrated_lattice.forward(torch.tensor([[1.0, 2.0]]))

        mock_calibrate_and_stack.assert_called_once()
        mock_lattice.assert_called_once()
        mock_output_calibrator.assert_called_once()


@pytest.mark.parametrize(
    "interpolation",
    [
        Interpolation.HYPERCUBE,
        Interpolation.SIMPLEX,
    ],
)
@pytest.mark.parametrize(
    "lattice_dim",
    [
        2,
        3,
        4,
    ],
)
def test_training_(interpolation, lattice_dim):  # pylint: disable=too-many-locals
    """Tests `CalibratedLattice` training on data from f(x) = 0.7|x_1| + 0.3x_2."""
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

    calibrated_lattice = CalibratedLattice(
        features=[
            NumericalFeature(
                "x1", x_1_numpy, num_keypoints=4, output_keypoints=lattice_dim
            ),
            CategoricalFeature(
                "x2",
                [0, 1, 2],
                monotonicity_pairs=[(0, 1), (1, 2)],
                output_keypoints=lattice_dim,
            ),
        ],
        output_min=output_min,
        output_max=output_max,
        interpolation=interpolation,
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(calibrated_lattice.parameters(recurse=True), lr=1e-1)

    with torch.no_grad():
        initial_predictions = calibrated_lattice(training_examples)
        initial_loss = loss_fn(initial_predictions, training_labels)

    train_calibrated_module(
        calibrated_lattice,
        training_examples,
        training_labels,
        loss_fn,
        optimizer,
        500,
        num_examples // 10,
    )

    with torch.no_grad():
        trained_predictions = calibrated_lattice(training_examples)
        trained_loss = loss_fn(trained_predictions, training_labels)

    # calibrated_lattice.constrain()
    print(calibrated_lattice.assert_constraints().items())
    assert trained_loss < initial_loss
    assert trained_loss < 0.08


def test_training_lattice_loan():
    """Tests `CalibratedLattice` on 5 features from loan approval data."""

    data = pd.read_csv(
        "https://github.com/SOTAI-Labs/datasets/raw/main/loan_approval.csv"
    )
    data["loan_status"] = data["loan_status"].apply(
        lambda x: 0 if x == "Rejected" else 1
    )
    csv_data = CSVData(data)
    batch_size = 512

    feature_configs = [
        NumericalFeature(
            "income_annum",
            data=csv_data("income_annum"),
            monotonicity=Monotonicity.INCREASING,
        ),
        NumericalFeature("loan_amount", csv_data("loan_amount")),
        NumericalFeature(
            "bank_asset_value",
            csv_data("bank_asset_value"),
            monotonicity=Monotonicity.INCREASING,
        ),
        NumericalFeature(
            "cibil_score", csv_data("cibil_score"), monotonicity=Monotonicity.INCREASING
        ),
        NumericalFeature("no_of_dependents", csv_data("no_of_dependents")),
        NumericalFeature("luxury_assets_value", csv_data("luxury_assets_value")),
        CategoricalFeature(
            "education",
            ["Not Graduate", "Graduate"],
            monotonicity_pairs=[("Not Graduate", "Graduate")],
        ),
        CategoricalFeature("self_employed", ["No", "Yes"]),
    ]

    csv_data.prepare(features=feature_configs, target_header="loan_status")
    training_examples_tensor, labels_tensor = next(csv_data.batch(batch_size))

    calibrated_lattice = CalibratedLattice(
        features=feature_configs, output_min=0.0, output_max=1.0
    )
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(calibrated_lattice.parameters(recurse=True), lr=1e-1)

    with torch.no_grad():
        initial_predictions = calibrated_lattice(training_examples_tensor)
        initial_loss = loss_fn(initial_predictions, labels_tensor)

    train_calibrated_module_tqdm(
        calibrated_module=calibrated_lattice,
        examples=csv_data,
        labels=labels_tensor,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=25,
        batch_size=batch_size,
    )

    with torch.no_grad():
        trained_predictions = calibrated_lattice(training_examples_tensor)
        trained_loss = loss_fn(trained_predictions, labels_tensor)

    assert trained_loss < initial_loss


def test_assert_constraints():
    """Tests `assert_constraints()` method calls internal assert_constraints."""
    calibrated_lattice = CalibratedLattice(
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
        output_calibration_num_keypoints=5,
    )

    with patch.object(
        calibrated_lattice.lattice, "assert_constraints", Mock()
    ) as mock_lattice_assert_constraints:
        mock_asserts = []
        for calibrator in calibrated_lattice.calibrators.values():
            mock_assert = Mock()
            calibrator.assert_constraints = mock_assert
            mock_asserts.append(mock_assert)
        mock_output_assert = Mock()
        calibrated_lattice.output_calibrator.assert_constraints = mock_output_assert

        calibrated_lattice.assert_constraints()

        mock_lattice_assert_constraints.assert_called_once()
        for mock_assert in mock_asserts:
            mock_assert.assert_called_once()
        mock_output_assert.assert_called_once()


def test_constrain():
    """Tests `constrain()` method calls internal constrain functions."""
    calibrated_lattice = CalibratedLattice(
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
        calibrated_lattice.lattice, "constrain", Mock()
    ) as mock_lattice_constrain:
        with patch.object(
            calibrated_lattice.output_calibrator, "constrain", Mock()
        ) as mock_output_calibrator_constrain:
            mock_constrains = []
            for calibrator in calibrated_lattice.calibrators.values():
                mock_calibrator_constrain = Mock()
                calibrator.constrain = mock_calibrator_constrain
                mock_constrains.append(mock_calibrator_constrain)

            calibrated_lattice.constrain()

            mock_lattice_constrain.assert_called_once()
            mock_output_calibrator_constrain.assert_called_once()
            for mock_constrain in mock_constrains:
                mock_constrain.assert_called_once()
