"""Random Tiny Lattice module for use in calibrated modeling.

PyTorch implementation of a RTL layer.
This layer takes several inputs which would otherwise be slow to run on a single lattice
and runs random subsets on an assortment of Random Tiny Lattices as an optimization.
"""
import logging
from typing import List, Optional

import numpy as np
import torch

from ..enums import Interpolation, LatticeInit, Monotonicity
from .lattice import Lattice


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
class RTL(torch.nn.Module):
    """An RTL Module.

    Layer takes a number of features that would otherwise be too many to assign to
    a single lattice, and instead assigns small random subsets of the features to an
    ensemble of smaller lattices. The features are shuffled and uniformly repeated
    if there are more slots in the RTL than features.

    Attributes:
      - All `__init__` arguments.
      - _lattice_layers: `dict` of form `{monotonic_count: (lattice, groups)}` which
      keeps track of the RTL structure. Features are indexed then randomly grouped
      together to be assigned to a lattice - groups with the same number of
      monotonic features can be put into the same lattice for further optimization,
      and are thus stored together in the dict according to `monotonic_count`.

    Example:
    `python
    inputs=torch.tensor(...) # shape: (batch_size, D)
    monotonicities = List[Monotonicity...] # len: D
    rtl=RTL(
      monotonicities,
      num_lattices = 5
      lattice_rank = 3, # num_lattices * lattice_rank must be greater than D
    )
    output = RTL(inputs)
    `
    """

    def __init__(
        self,
        monotonicities: List[Monotonicity],
        num_lattices: int,
        lattice_rank: int,
        lattice_size: int = 2,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        kernel_init: LatticeInit = LatticeInit.LINEAR,
        clip_inputs: bool = True,
        interpolation: Interpolation = Interpolation.HYPERCUBE,
        average_outputs: bool = False,
        random_seed: int = 42,
    ) -> None:
        """Initializes an instance of 'RTL'.

        Args:
            monotonicities: `List` of Monotonicity.INCREASING or Monotonicity.NONE
             indicating monotonicities of input features, ordered respectively.
            num_lattices: number of lattices in RTL structure.
            lattice_rank: number of inputs for each lattice in RTL structure.
            output_min: Minimum output of each lattice in RTL.
            output_max: Maximum output of each lattice in RTL.
            kernel_init: Initialization scheme to use for lattices.
            clip_inputs: Whether input should be clipped to the range of each lattice.
            interpolation: Interpolation scheme for each lattice in RTL.
            average_outputs: Whether to average the outputs of every lattice RTL.
            random_seed: seed used for shuffling.

        Raises:
            ValueError: if size of RTL, determined by `num_lattices * lattice_rank`, is
             too small to support the number of input features.
        """
        super().__init__()

        if len(monotonicities) > num_lattices * lattice_rank:
            raise ValueError(
                f"RTL with {num_lattices}x{lattice_rank}D structure cannot support "
                + f"{len(monotonicities)} input features."
            )
        self.monotonicities = monotonicities
        self.num_lattices = num_lattices
        self.lattice_rank = lattice_rank
        self.lattice_size = lattice_size
        self.output_min = output_min
        self.output_max = output_max
        self.kernel_init = kernel_init
        self.clip_inputs = clip_inputs
        self.interpolation = interpolation
        self.average_outputs = average_outputs
        self.random_seed = random_seed

        rtl_indices = np.array(
            [i % len(self.monotonicities) for i in range(num_lattices * lattice_rank)]
        )
        np.random.seed(self.random_seed)
        np.random.shuffle(rtl_indices)
        rtl_indices = np.split(rtl_indices, num_lattices)
        rtl_indices = self._ensure_unique_sublattices(rtl_indices)
        monotonicity_groupings = {}
        for lattice_indices in rtl_indices:
            monotonic_count = sum(
                1
                for idx in lattice_indices
                if self.monotonicities[idx] == Monotonicity.INCREASING
            )
            if monotonic_count not in monotonicity_groupings:
                monotonicity_groupings[monotonic_count] = [lattice_indices]
            else:
                monotonicity_groupings[monotonic_count].append(lattice_indices)
        for monotonic_count, groups in monotonicity_groupings.items():
            for i, lattice_indices in enumerate(groups):
                sorted_indices = sorted(
                    lattice_indices,
                    key=lambda x: (self.monotonicities[x] != "increasing"),
                    reverse=False,
                )
                groups[i] = sorted_indices

        self._lattice_layers = {}
        for monotonic_count, groups in monotonicity_groupings.items():
            self._lattice_layers[monotonic_count] = (
                Lattice(
                    lattice_sizes=[self.lattice_size] * self.lattice_rank,
                    output_min=self.output_min,
                    output_max=self.output_max,
                    kernel_init=self.kernel_init,
                    monotonicities=[Monotonicity.INCREASING] * monotonic_count
                    + [Monotonicity.NONE] * (lattice_rank - monotonic_count),
                    clip_inputs=self.clip_inputs,
                    interpolation=self.interpolation,
                    units=len(groups),
                ),
                groups,
            )

    # pylint: disable-next=invalid-name
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method computed by using forward methods of each lattice in RTL.

        Args:
            x: input tensor of feature values with shape `(batch_size, num_features)`.

        Returns:
            torch.Tensor containing the outputs of each lattice within RTL structure. If
            `average_outputs == True`, then all outputs are averaged into a tensor of
            shape `(batch_size, 1)`. If `average_outputs == False`, shape of tensor is
            `(batch_size, num_lattices)`.
        """
        result = []
        for _, (lattice, group) in sorted(self._lattice_layers.items()):
            if len(group) > 1:
                lattice_input = torch.stack([x[:, idx] for idx in group], dim=-2)
            else:
                lattice_input = x[:, group[0]]
            result.append(lattice.forward(lattice_input))
        result = torch.cat(result, dim=-1)
        if not self.average_outputs:
            return result
        result = torch.mean(result, dim=-1, keepdim=True)

        return result

    @torch.no_grad()
    def constrain(self) -> None:
        """Enforces constraints for each lattice in RTL."""
        for lattice, _ in self._lattice_layers.values():
            lattice.constrain()

    @torch.no_grad()
    def assert_constraints(self, eps=1e-6) -> List[List[str]]:
        """Asserts that each Lattice in RTL satisfies all constraints.

        Args:
          eps: allowed constraints violations.

        Returns:
          List of lists, each with constraints violations for an individual Lattice.
        """
        return list(
            lattice.assert_constraints(eps=eps)
            for lattice, _ in self._lattice_layers.values()
        )

    @staticmethod
    def _ensure_unique_sublattices(
        rtl_indices: np.ndarray,
        max_swaps: int = 10000,
    ) -> np.ndarray:
        """Attempts to ensure every lattice in RTL structure contains unique features.

        Args:
            rtl_indices: 2-D `numpy.ndarray` where inner lists are groupings of indices
                of input features to RTL layer.
            max_swaps: maximum number of swaps to perform before giving up.

        Returns:
            2-D `numpy.ndarray` where elements between inner lists have been swapped in
            an attempt to remove any duplicates from every grouping.
        """
        swaps = 0
        num_sublattices = len(rtl_indices)

        def find_swap_candidate(current_index, element):
            """Helper function to find the next sublattice not containing element."""
            for offset in range(1, num_sublattices):
                candidate_index = (current_index + offset) % num_sublattices
                if element not in rtl_indices[candidate_index]:
                    return candidate_index
            return None

        # pylint: disable-next=R1702
        for i, sublattice in enumerate(rtl_indices):
            unique_elements = set()
            for element in sublattice:
                if element in unique_elements:
                    swap_with = find_swap_candidate(i, element)
                    if swap_with is not None:
                        for swap_element in rtl_indices[swap_with]:
                            if swap_element not in sublattice:
                                idx_to_swap = np.nonzero(
                                    rtl_indices[swap_with] == swap_element
                                )[0][0]
                                idx_duplicate = np.nonzero(sublattice == element)[0][0]
                                (
                                    rtl_indices[swap_with][idx_to_swap],
                                    sublattice[idx_duplicate],
                                ) = (element, swap_element)
                                swaps += 1
                                break
                    else:
                        logging.info(
                            "Some lattices in RTL may use the same feature multiple "
                            "times."
                        )
                        return rtl_indices
                else:
                    unique_elements.add(element)
                if swaps >= max_swaps:
                    logging.info(
                        "Some lattices in RTL may use the same feature multiple times."
                    )
                    return rtl_indices
        return rtl_indices
