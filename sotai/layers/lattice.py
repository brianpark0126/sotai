"""Lattice module for use in calibrated modeling.

PyTorch implementation of a lattice layer.
This layer takes one or more d-dimensional inputs and outputs the interpolated value
according the specified interpolation method.
"""
import torch
import numpy as np
from typing import Iterator, Tuple, Union, List


class Lattice(torch.nn.Module):
    """A Lattice Module.

    Layer perfors interpolation using one of 'units' d-dimensional lattices with
    arbitrary number of keypoints per dimension. Each lattice vertex has a trainable
    weight, and input is considered to be a d-dimensional point within the lattice.
    """

    def __init__(
        self,
        lattice_sizes,
        clip_inputs=True,
        interpolation="hypercube",
        units=1,
    ) -> None:
        super().__init__()
        self.clip_inputs = clip_inputs
        self.lattice_sizes = lattice_sizes
        self.interpolation = interpolation
        self.units = units
        self.kernel = torch.nn.Parameter(
            torch.randn(np.prod(lattice_sizes), units).double()
        )

    def forward(self, x) -> torch.Tensor:
        """Calculates interpolation from input, using method of self.interpolation.

        Args:
            x: input tensor. If units == 1, tensor of shape: (batch_size, ...,
              len(lattice_size)) or list of len(lattice_sizes) tensors of same shape:
              (batch_size, ..., 1). If units > 1, tensor of shape (batch_size, ...,
              units, len(lattice_sizes)) or list of len(lattice_sizes) tensors of same
              shape (batch_size, ..., units, 1)

        Returns:
            torch.Tensor of shape (batch_size, ..., units) containing interpolated
              values.
        """
        if self.interpolation == "hypercube":
            return self._compute_hypercube_interpolation(x.double())
        else:
            """TODO"""
            return None

    def _compute_hypercube_interpolation(self, inputs) -> torch.Tensor:
        """Performs hypercube interpolation

        Args:
            inputs: input tensor. If units == 1, tensor of shape: (batch_size, ...,
              len(lattice_size)) or list of len(lattice_sizes) tensors of same shape:
              (batch_size, ..., 1). If units > 1, tensor of shape (batch_size, ...,
              units, len(lattice_sizes)) or list of len(lattice_sizes) tensors of same
              shape (batch_size, ..., units, 1)

        Returns:
            torch.Tensor of shape (batch_size, ..., units) containing interpolated
              values.
        """
        interpolation_weights = self._compute_interpolation_weights(
            inputs=inputs, clip_inputs=self.clip_inputs
        )
        if self.units == 1:
            return torch.matmul(interpolation_weights, self.kernel)
        else:
            return torch.sum(interpolation_weights * self.kernel.t(), dim=-1)

    def _compute_interpolation_weights(self, inputs, clip_inputs=True) -> torch.Tensor:
        """Computes weights for hypercube lattice interpolation.

        Weights will be a matrix consisting of zeros and the outer product of (1-x_i,
        x_i) for all x_i in x (in the proper order). This matrix can be multiplied by
        the lattice layer's kernel to compute final interpolation value.

        Args:
            inputs: torch.Tensor of shape (batch_size, ..., len(lattice_sizes) or list
              of len(lattice_sizes) tensors of same shape (batch_size, ..., 1)

            clip_inputs: Boolean to determine whether input values outside lattice
              bounds should be clipped to the min or max supported values.

        Returns:
            torch.Tensor of shape (batch_size, ..., prod(lattice_sizes)) containing the
              weights which can be matrix multiplied with the kernel to perform
              hypercube interpolation.
        """
        if isinstance(inputs, list):
            input_dtype = inputs[0].dtype
        else:
            input_dtype = inputs.dtype

        # Special case: 2^d lattice with input passed in as a single tensor
        if all(size == 2 for size in self.lattice_sizes) and not isinstance(
            inputs, list
        ):
            w = torch.stack([(1.0 - inputs), inputs], dim=-1)
            if clip_inputs:
                w = torch.clamp(w, min=0, max=1)
            one_d_interpolation_weights = list(torch.unbind(w, dim=-2))
            return self._batch_outer_operation(one_d_interpolation_weights)

        if clip_inputs:
            inputs = self._clip_onto_lattice_range(inputs)

        dim_keypoints = {}
        for dim_size in set(self.lattice_sizes):
            dim_keypoints[dim_size] = torch.tensor(
                [i for i in range(dim_size)], dtype=input_dtype
            )
        bucketized_inputs = self._bucketize_consecutive_equal_dims(inputs)
        one_d_interpolation_weights = []

        for tensor, bucket_size, dim_size in bucketized_inputs:
            if bucket_size > 1:
                tensor = torch.unsqueeze(tensor, dim=-1)
            distance = torch.abs(tensor - dim_keypoints[dim_size])
            weights = 1.0 - torch.minimum(
                distance, torch.tensor(1.0, dtype=distance.dtype)
            )
            if bucket_size == 1:
                one_d_interpolation_weights.append(weights)
            else:
                one_d_interpolation_weights.extend(torch.unbind(weights, dim=-2))

        return self._batch_outer_operation(one_d_interpolation_weights)

    @staticmethod
    def _batch_outer_operation(list_of_tensors) -> torch.Tensor:
        """Computes the outer product of a list of tensors.

        Args:
            list_of_tensors: List of tensors of same shape (batch_size, ..., k[i])
              where everything except `k_i` matches.

        Returns:
            torch.Tensor of shape (batch_size, ..., k_i * k_j * ...) containing a
              flattened version of the outer product.
        """
        if len(list_of_tensors) == 1:
            return list_of_tensors[0]

        result = torch.unsqueeze(list_of_tensors[0], dim=-1)

        for i, tensor in enumerate(list_of_tensors[1:]):
            op = torch.mul if i < 6 else torch.matmul
            result = op(result, torch.unsqueeze(tensor, dim=-2))
            shape = [-1] + [int(size) for size in result.shape[1:]]
            new_shape = shape[:-2] + [shape[-2] * shape[-1]]
            if i < len(list_of_tensors) - 2:
                new_shape.append(1)
            result = torch.reshape(result, new_shape)

        return result

    def _clip_onto_lattice_range(self, inputs) -> torch.Tensor:
        """Clips inputs onto valid input range for given lattice_sizes.

        Args:
            inputs: "inputs" argument of _compute_interpolation_weights()

        Returns:
            torch.Tensor of shape "inputs" with va
        """
        if not isinstance(inputs, list):
            upper_bounds = torch.tensor(
                [dim_size - 1.0 for dim_size in self.lattice_sizes]
            ).double()
            clipped_inputs = torch.clamp(
                inputs, min=torch.zeros_like(upper_bounds), max=upper_bounds
            )
        else:
            dim_upper_bounds = {}
            for dim_size in set(self.lattice_sizes):
                dim_upper_bounds[dim_size] = torch.tensor(
                    dim_size - 1.0, dtype=inputs[0].dtype
                )
            dim_lower_bound = torch.zeros(1, dtype=inputs[0].dtype)

            clipped_inputs = []
            for one_d_input, dim_size in zip(inputs, self.lattice_sizes):
                clipped_inputs.append(
                    torch.clamp(
                        one_d_input, min=dim_lower_bound, max=dim_upper_bounds[dim_size]
                    )
                )

        return clipped_inputs

    def _bucketize_consecutive_equal_dims(
        self, inputs
    ) -> Iterator[Tuple[torch.Tensor, int, int]]:
        """Creates buckets of equal sized dimensions for broadcasting ops.

        Args:
            inputs: "inputs" argument of _compute_interpolation_weights().

        Returns:
            An Iterable containing (torch.Tensor, int, int) where the Tensor contains
            individual values from "inputs" corresponding to its bucket, the first int
            is bucket size, and the second int is size of the dimension of the bucket.

        """
        if not isinstance(inputs, list):
            bucket_sizes = []
            bucket_dim_sizes = []
            current_size = 1
            for i in range(1, len(self.lattice_sizes)):
                if self.lattice_sizes[i] != self.lattice_sizes[i - 1]:
                    bucket_sizes.append(current_size)
                    bucket_dim_sizes.append(self.lattice_sizes[i - 1])
                    current_size = 1
                else:
                    current_size += 1
            bucket_sizes.append(current_size)
            bucket_dim_sizes.append(self.lattice_sizes[-1])
            inputs = torch.split(inputs, split_size_or_sections=bucket_sizes, dim=-1)
        else:
            bucket_sizes = [1] * len(self.lattice_sizes)
            bucket_dim_sizes = self.lattice_sizes

        return zip(inputs, bucket_sizes, bucket_dim_sizes)
