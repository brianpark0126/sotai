<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/lattice.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.layers.lattice`
Lattice module for use in calibrated modeling. 

PyTorch implementation of a lattice layer. This layer takes one or more d-dimensional inputs and outputs the interpolated value according the specified interpolation method. 



---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/lattice.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Lattice`
A Lattice Module. 

Layer performs interpolation using one of 'units' d-dimensional lattices with arbitrary number of keypoints per dimension. Each lattice vertex has a trainable weight, and input is considered to be a d-dimensional point within the lattice. 



**Attributes:**
 
  - All `__init__` arguments. 
 - <b>`kernel`</b>:  `torch.nn.Parameter` of shape `(prod(lattice_sizes), units)` which stores  weights at each vertex of lattice. 



**Example:**
 `python lattice_sizes = [2, 2, 4, 3] inputs=torch.tensor(...) # shape: (batch_size, len(lattice_sizes)) lattice=Lattice( lattice_sizes, clip_inputs=True, interpolation=Interpolation.Hypercube, units=1, ) outputs = Lattice(inputs) ` 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/lattice.py#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    lattice_sizes: Union[List[int], Tuple[int]],
    output_min: float = 0.0,
    output_max: float = 1.0,
    kernel_init: LatticeInit = 'linear',
    clip_inputs: bool = True,
    interpolation: Interpolation = 'hypercube',
    units: int = 1
) → None
```

Initializes an instance of 'Lattice'. 



**Args:**
 
 - <b>`lattice_sizes`</b>:  List or tuple of size of lattice along each dimension. 
 - <b>`output_min`</b>:  Minimum output value for weights at vertices of lattice. 
 - <b>`output_max`</b>:  Maximum output value for weights at vertices of lattice. 
 - <b>`kernel_init`</b>:  Initialization scheme to use for the kernel. 
 - <b>`clip_inputs`</b>:  Whether input points should be clipped to the range of lattice. 
 - <b>`interpolation`</b>:  Interpolation scheme for a given input. 
 - <b>`units`</b>:  Dimensionality of weights stored at each vertex of lattice. 



**Raises:**
 
 - <b>`ValueError`</b>:  if `kernel_init` is invalid. 
 - <b>`NotImplementedError`</b>:  Random monotonic initialization not yet implemented. 




---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/lattice.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Union[Tensor, List[Tensor]]) → Tensor
```

Calculates interpolation from input, using method of self.interpolation. 



**Args:**
 
 - <b>`x`</b>:  input tensor. If `units == 1`, tensor of shape:  `(batch_size, ..., len(lattice_size))` or list of `len(lattice_sizes)` 
 - <b>`tensors of same shape`</b>:  `(batch_size, ..., 1)`. If `units > 1`, tensor of shape `(batch_size, ..., units, len(lattice_sizes))` or list of `len(lattice_sizes)` tensors of same shape `(batch_size, ..., units, 1)`. 



**Returns:**
 torch.Tensor of shape `(batch_size, ..., units)` containing interpolated values. 



**Raises:**
 
 - <b>`NotImplementedError`</b>:  If `interpolation == simplex`, as yet not implemented. 
 - <b>`ValueError`</b>:  If the type of interpolation is unknown. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
