"""Random Tiny Lattice module for use in calibrated modeling.

PyTorch implementation of a RTL layer.
This layer takes one or more d-dimensional inputs and outputs the interpolated value
according the specified interpolation method.
"""
import collections
import itertools
import logging
from typing import Iterator, Tuple, Union, List, Callable, Optional

import numpy as np
import torch

from ..enums import Interpolation, LatticeInit, Monotonicity
from lattice import Lattice

_MAX_RTL_SWAPS = 10000
_RTLInput = collections.namedtuple('_RTLInput',
                                   ['monotonicity', 'group', 'input_index'])
RTL_KFL_NAME = 'rtl_kronecker_factored_lattice'
RTL_LATTICE_NAME = 'rtl_lattice'
INPUTS_FOR_UNITS_PREFIX = 'inputs_for_lattice'
RTL_CONCAT_NAME = 'rtl_concat'


class RTL(torch.nn.Module):

    def __init__(
            self,
            num_lattices: int,
            lattice_rank: int,
            lattice_size: int = 2,
            output_min: Optional[float] = None,
            output_max: Optional[float] = None,
            kernel_init: LatticeInit = LatticeInit.LINEAR,
            clip_inputs: bool = True,
            interpolation: Interpolation = Interpolation.HYPERCUBE,
            separate_outputs: bool = False,
            random_seed: int = 42,
            parametrization: str = "all_vertices",
            avoid_intragroup_interaction: bool = True,
            average_outputs: bool = False,
    ) -> None:
        self.num_lattices = num_lattices
        self.lattice_rank = lattice_rank
        self.lattice_size = lattice_size
        self.output_min = output_min
        self.output_max = output_max
        self.kernel_init = kernel_init
        self.clip_inputs = clip_inputs
        self.interpolation = interpolation
        self.separate_outputs = separate_outputs
        self.random_seed = random_seed
        self.parametrizatoin = parametrization
        self.avoid_intragroup_interaction = avoid_intragroup_interaction
        self.average_outputs = average_outputs

        self._lattice_layers = {}

    def forward(self, x) -> torch.Tensor:
        # Check if x is a dictionary. If not, make it one.
        if not isinstance(x, dict):
            x = {'unconstrained': x}

        # Flatten the input.
        input_tensors = []
        for input_key in sorted(x.keys()):
            items = x[input_key]
            if isinstance(items, list):
                input_tensors.extend(items)
            else:
                input_tensors.append(items)

        if len(input_tensors) == 1:
            flattened_input = input_tensors[0]
        else:
            flattened_input = torch.cat(input_tensors, dim=1)

        outputs_for_monotonicity = [[], []]
        for monotonicities, inputs_for_units in self._rtl_structure:
            if len(inputs_for_units) == 1:
                inputs_for_units = inputs_for_units[0]
            lattice_inputs = flattened_input.index_select(1, torch.tensor(
                inputs_for_units))
            output_monotonicity = max(monotonicities)
            outputs_for_monotonicity[output_monotonicity].append(
                self._lattice_layers[str(monotonicities)](lattice_inputs))

        if self.separate_outputs:
            separate_outputs = {}
            for monotoncity, output_key in [(0, 'unconstrained'), (1, 'increasing')]:
                lattice_outputs = outputs_for_monotonicity[monotoncity]
                if not lattice_outputs:
                    pass
                elif len(lattice_outputs) == 1:
                    separate_outputs[output_key] = lattice_outputs[0]
                else:
                    separate_outputs[output_key] = torch.cat(lattice_outputs, dim=1)
            return separate_outputs
        else:
            joint_outputs = outputs_for_monotonicity[0] + outputs_for_monotonicity[1]
            if len(joint_outputs) > 1:
                joint_outputs = torch.cat(joint_outputs, dim=1)
            else:
                joint_outputs = joint_outputs[0]
            if self.average_outputs:
                joint_outputs = torch.mean(joint_outputs, dim=-1, keepdim=True)
            return joint_outputs

    @torch.no_grad()
    def constrain(self) -> None:
        """Ensures layers weights strictly satisfy constraints.

        Applies approximate projection to strictly satisfy specified constraints.
        If `monotonic_at_every_step == True` there is no need to call this function.

        Returns:
          In eager mode directly updates weights and returns variable which stores
          them. In graph mode returns a list of `assign_add` op which has to be
          executed to updates weights.
        """
        for lattice in self._lattice_layers.values():
            lattice.constrain()

    @torch.no_grad()
    def assert_constraints(self, eps=1e-6) -> List[List[str]]:
        """Asserts that weights satisfy all constraints.

        In graph mode builds and returns a list of assertion ops.
        In eager mode directly executes assertions.

        Args:
          eps: allowed constraints violation.

        Returns:
          List of assertion ops in graph mode or immediately asserts in eager mode.
        """
        return list(lattice.assert_constraints(eps=eps) for lattice in self._lattice_layers.values())

    @torch.no_grad()
    def _get_rtl_structure(self, input_shape):
        """Returns the RTL structure for the given input_shape.

        Args:
            input_shape: Input shape to the layer. Must be a dict matching the format
            described in the layer description.

        Raises:
          ValueError: If the structure is too small to include all the inputs.

        Returns:
          A list of `(monotonicities, lattices)` tuples, where `monotonicities` is
          the tuple of lattice monotonicites, and `lattices` is a list of list of
          indices into the flattened input to the layer.
        """

        # Convert to dict if not already
        if not isinstance(input_shape, dict):
            input_shape = {'unconstrained': input_shape}

        rtl_inputs = []
        group = 0
        input_index = 0
        for input_key in sorted(input_shape.keys()):
            shapes = input_shape[input_key]
            if input_key == 'unconstrained':
                monotonicity = 0
            elif input_key == 'increasing':
                monotonicity = 1
            else:
                raise ValueError(
                    f'Unrecognized key in the input to the RTL layer: {input_key}')

            if not isinstance(shapes, list):
                shapes = [(shapes[0], 1)] * shapes[1]

            for shape in shapes:
                for _ in range(shape[1]):
                    rtl_inputs.append(
                        _RTLInput(monotonicity=monotonicity, group=group,
                                  input_index=input_index)
                    )
                    input_index += 1
                group += 1

        total_usage = self.num_lattices * self.lattice_rank
        if total_usage < len(rtl_inputs):
            raise ValueError(
                f'RTL layer with {self.num_lattices}x{self.lattice_rank}D lattices is too small to use all the {len(rtl_inputs)} input features')

        rs = np.random.RandomState(self.random_seed)
        rtl_inputs = np.array(rtl_inputs)  # Convert list to array for shuffling
        rs.shuffle(rtl_inputs)
        rtl_inputs = list(rtl_inputs) * (1 + total_usage // len(rtl_inputs))
        rtl_inputs = rtl_inputs[:total_usage]
        rs.shuffle(rtl_inputs)

        lattices = [rtl_inputs[i * self.lattice_rank: (i + 1) * self.lattice_rank] for i
                    in range(self.num_lattices)]

        changed = True
        iteration = 0
        while changed and self.avoid_intragroup_interaction:
            if iteration > _MAX_RTL_SWAPS:
                logging.info('Some lattices in the RTL layer might use features from '
                             'the same input group')
                break
            changed = False
            iteration += 1
            for lattice_0, lattice_1 in itertools.combinations(lattices, 2):
                # For every pair of lattices: lattice_0, lattice_1
                for index_0, index_1 in itertools.product(
                        range(len(lattice_0)), range(len(lattice_1))):
                    # Consider swapping lattice_0[index_0] with lattice_1[index_1]
                    rest_lattice_0 = list(lattice_0)
                    rest_lattice_1 = list(lattice_1)
                    feature_0 = rest_lattice_0.pop(index_0)
                    feature_1 = rest_lattice_1.pop(index_1)
                    if feature_0.group == feature_1.group:
                        continue

                    # Swap if a group is repeated and a swap fixes it.
                    rest_lattice_groups_0 = list(
                        lattice_input.group for lattice_input in rest_lattice_0)
                    rest_lattice_groups_1 = list(
                        lattice_input.group for lattice_input in rest_lattice_1)
                    if ((feature_0.group in rest_lattice_groups_0) and
                            (feature_0.group not in rest_lattice_groups_1) and
                            (feature_1.group not in rest_lattice_groups_0)):
                        lattice_0[index_0], lattice_1[index_1] = (lattice_1[index_1],
                                                                  lattice_0[index_0])
                        changed = True

        # Arrange into combined lattices layers. Lattices with similar monotonicites
        # can use the same tfl.layers.Lattice layer.
        # Create a dict: monotonicity -> list of list of input indices.
        lattices_for_monotonicities = collections.defaultdict(list)
        for lattice in lattices:
            lattice.sort(key=lambda lattice_input: lattice_input.monotonicity)
            monotonicities = tuple(
                lattice_input.monotonicity for lattice_input in lattice)
            lattice_input_indices = list(
                lattice_input.input_index for lattice_input in lattice)
            lattices_for_monotonicities[monotonicities].append(lattice_input_indices)

        return sorted(lattices_for_monotonicities.items())