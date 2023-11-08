"""Tests for RTL layer."""
from unittest.mock import patch, Mock
import torch
import pytest
import numpy as np

from sotai import Monotonicity, LatticeInit, Interpolation
from sotai.layers import RTL
from sotai.layers import Lattice
# pylint: disable=invalid-name
# pylint: disable=protected-access

@pytest.mark.parametrize(
    "monotonicities, num_lattices, lattice_rank, output_min, output_max, kernel_init,"
    "clip_inputs, interpolation, average_outputs",
    [
        (
            [
                Monotonicity.NONE,
                Monotonicity.NONE,
                Monotonicity.NONE,
                Monotonicity.NONE,
            ],
            3,
            3,
            None,
            2.0,
            LatticeInit.LINEAR,
            True,
            Interpolation.HYPERCUBE,
            True,
        ),
        (
            [
                Monotonicity.INCREASING,
                Monotonicity.INCREASING,
                Monotonicity.NONE,
                Monotonicity.NONE,
            ],
            3,
            3,
            -1.0,
            4.0,
            LatticeInit.LINEAR,
            False,
            Interpolation.SIMPLEX,
            False,
        ),
        (
            [Monotonicity.INCREASING, Monotonicity.NONE] * 25,
            20,
            5,
            None,
            None,
            LatticeInit.LINEAR,
            True,
            Interpolation.HYPERCUBE,
            True,
        ),
    ],
)
# pylint: disable-next=too-many-locals
def test_initialization(
    monotonicities,
    num_lattices,
    lattice_rank,
    output_min,
    output_max,
    kernel_init,
    clip_inputs,
    interpolation,
    average_outputs,
):
    """Tests that RTL Initialization works properly."""
    rtl = RTL(
        monotonicities=monotonicities,
        num_lattices=num_lattices,
        lattice_rank=lattice_rank,
        output_min=output_min,
        output_max=output_max,
        kernel_init=kernel_init,
        clip_inputs=clip_inputs,
        interpolation=interpolation,
        average_outputs=average_outputs,
    )
    assert rtl.monotonicities == monotonicities
    assert rtl.num_lattices == num_lattices
    assert rtl.lattice_rank == lattice_rank
    assert rtl.output_min == output_min
    assert rtl.output_max == output_max
    assert rtl.kernel_init == kernel_init
    assert rtl.interpolation == interpolation
    assert rtl.average_outputs == average_outputs

    total_lattices = 0
    for monotonic_count, (lattice, group) in rtl._lattice_layers.items():
        # test monotonic features have been sorted to front of list for lattice indices
        for single_lattice_indices in group:
            for i in range(lattice_rank):
                if i < monotonic_count:
                    assert (
                        rtl.monotonicities[single_lattice_indices[i]]
                        == Monotonicity.INCREASING
                    )
                else:
                    assert (
                        rtl.monotonicities[single_lattice_indices[i]]
                        == Monotonicity.NONE
                    )

        assert len(lattice.monotonicities) == len(lattice.lattice_sizes)
        assert (
            sum(1 for _ in lattice.monotonicities if _ == Monotonicity.INCREASING)
            == monotonic_count
        )
        assert lattice.output_min == rtl.output_min
        assert lattice.output_max == rtl.output_max
        assert lattice.kernel_init == rtl.kernel_init
        assert lattice.clip_inputs == rtl.clip_inputs
        assert lattice.interpolation == rtl.interpolation

        # test number of lattices created is correct
        total_lattices += lattice.units

    assert total_lattices == num_lattices


@pytest.mark.parametrize(
    "monotonicities, num_lattices, lattice_rank",
    [
        ([Monotonicity.NONE] * 9, 2, 2),
        ([Monotonicity.INCREASING] * 10, 3, 3),
    ],
)
def test_initialization_invalid(
    monotonicities,
    num_lattices,
    lattice_rank,
):
    """Tests that RTL Initialization raises error when RTL is too small."""
    with pytest.raises(ValueError) as exc_info:
        RTL(
            monotonicities=monotonicities,
            num_lattices=num_lattices,
            lattice_rank=lattice_rank,
        )
    assert (
        str(exc_info.value)
        == f"RTL with {num_lattices}x{lattice_rank}D structure cannot support "
        + f"{len(monotonicities)} input features."
    )


def test_forward():
    """Tests forward function of RTL layer."""
    # num features = 6
    # num_lattices = 6
    # lattice_rank = 3
    rtl = RTL(
        monotonicities=[Monotonicity.NONE, Monotonicity.INCREASING] * 3,
        num_lattices=6,
        lattice_rank=3,
    )
    rtl._lattice_layers = {
        0: (
            Lattice(lattice_sizes=[2, 2, 2], units=3),
            [[0, 1, 2], [3, 4, 5], [0, 2, 4]],
        ),
        1: (Lattice(lattice_sizes=[2, 2, 2], units=2), [[1, 3, 5], [1, 2, 3]]),
        2: (Lattice(lattice_sizes=[2, 2, 2], units=1), [[4, 5, 0]]),
    }
    x = torch.tensor(
        [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.01, 0.11, 0.21, 0.31, 0.41, 0.51]]
    )
    units = [3, 2, 1]
    batch_size = 2
    mock_forwards = [None] * 3
    for monotonic_count, (lattice, _) in rtl._lattice_layers.items():
        mock_forward = Mock()
        lattice.forward = mock_forward
        mock_forward.return_value = torch.full(
            (batch_size, units[monotonic_count]),
            float(monotonic_count),
            dtype=torch.float,
        )
        mock_forwards[monotonic_count] = mock_forward

    result = rtl.forward(x)

    mock_forwards[0].assert_called_once()
    torch.allclose(
        torch.tensor(
            [
                [[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.0, 0.2, 0.4]],
                [[0.01, 0.11, 0.21], [0.31, 0.41, 0.51], [0.01, 0.21, 0.41]],
            ]
        ),
        mock_forwards[0].call_args[0][0],
    )
    mock_forwards[1].assert_called_once()
    torch.allclose(
        torch.tensor(
            [
                [[0.1, 0.3, 0.5], [0.1, 0.2, 0.3]],
                [[0.11, 0.31, 0.51], [0.11, 0.21, 0.31]],
            ]
        ),
        mock_forwards[1].call_args[0][0],
    )
    mock_forwards[2].assert_called_once()
    torch.allclose(
        torch.tensor([[0.4, 0.5, 0.0], [0.41, 0.51, 0.01]]),
        mock_forwards[2].call_args[0][0],
    )
    assert torch.allclose(
        result,
        torch.tensor([[0, 0, 0, 1, 1, 2], [0, 0, 0, 1, 1, 2]], dtype=torch.float),
    )

    rtl.average_outputs = True
    result = rtl.forward(x)
    assert torch.allclose(result, torch.tensor([[2 / 3], [2 / 3]], dtype=torch.float))


def test_constrain():
    """Tests RTL constrain function."""
    rtl = RTL(
        monotonicities=[Monotonicity.NONE, Monotonicity.INCREASING],
        num_lattices=3,
        lattice_rank=3
    )
    mock_constrains = []
    for lattice, _ in rtl._lattice_layers.values():
        mock_constrain = Mock()
        lattice.constrain = mock_constrain
        mock_constrains.append(mock_constrain)

    rtl.constrain()
    for mock_constrain in mock_constrains:
        mock_constrain.assert_called_once()


def test_assert_constraints():
    """Tests RTL assert_constraints function."""
    rtl = RTL(
        monotonicities=[Monotonicity.NONE, Monotonicity.INCREASING],
        num_lattices=3,
        lattice_rank=3
    )
    mock_asserts = []
    for lattice, _ in rtl._lattice_layers.values():
        mock_assert = Mock()
        lattice.assert_constraints = mock_assert
        mock_assert.return_value = "violation"
        mock_asserts.append(mock_assert)

    violations = rtl.assert_constraints()
    for mock_assert in mock_asserts:
        mock_assert.assert_called_once()

    assert violations == ["violation"] * len(rtl._lattice_layers)



@pytest.mark.parametrize(
    "rtl_indices",
    [
        [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [6, 6]],
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        [[1, 1, 1], [2, 2, 2], [1, 2, 3], [3, 3, 3]],
        [
            [1, 1, 2],
            [2, 3, 4],
            [1, 5, 5],
            [4, 6, 7],
            [1, 3, 4],
            [2, 3, 3],
            [4, 5, 6],
            [6, 6, 6],
        ],
    ],
)
def test_ensure_unique_sublattices_possible(rtl_indices):
    """Tests _ensure_unique_sublattices removes duplicates from groups when possible."""
    swapped_indices = RTL._ensure_unique_sublattices(np.array(rtl_indices))
    for group in swapped_indices:
        assert len(set(group)) == len(group)


@pytest.mark.parametrize(
    "rtl_indices, max_swaps",
    [
        ([[1, 1], [1, 2], [1, 3]], 100),
        ([[1, 1], [2, 2], [3, 3], [4, 4]], 2),
    ],
)
def test_ensure_unique_sublattices_impossible(rtl_indices, max_swaps):
    """Tests _ensure_unique_sublattices logs when it can't remove duplicates."""
    with patch("logging.info") as mock_logging_info:
        RTL._ensure_unique_sublattices(
            np.array(rtl_indices), max_swaps
        )
        mock_logging_info.assert_called_with(
            "Some lattices in RTL may use the same feature multiple times."
        )
