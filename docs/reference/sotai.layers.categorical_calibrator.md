<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/categorical_calibrator.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.layers.categorical_calibrator`
Categorical calibration module. 

PyTorch implementation of the categorical calibration module. This module takes in a single-dimensional input of categories represented as indices and transforms it by mapping a given category to its learned output value. 



---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/categorical_calibrator.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CategoricalCalibrator`
A categorical calibrator. 

This module takes an input of shape `(batch_size, 1)` and calibrates it by mapping a given category to its learned output value. The output will have the same shape as the input. 



**Attributes:**
 
    - All `__init__` arguments. 
 - <b>`kernel`</b>:  `torch.nn.Parameter` that stores the categorical mapping weights. 



**Example:**
 

```python
inputs = torch.tensor(...)  # shape: (batch_size, 1)
calibrator = CategoricalCalibrator(
    num_categories=5,
    missing_input_value=-1,
    output_min=0.0
    output_max=1.0,
    monotonicity_pairs=[(0, 1), (1, 2)],
    kernel_init=CateegoricalCalibratorInit.UNIFORM,
)
outputs = calibrator(inputs)
``` 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/categorical_calibrator.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    num_categories: int,
    missing_input_value: Optional[float] = None,
    output_min: Optional[float] = None,
    output_max: Optional[float] = None,
    monotonicity_pairs: Optional[List[Tuple[int, int]]] = None,
    kernel_init: CategoricalCalibratorInit = 'uniform'
) → None
```

Initializes an instance of `CategoricalCalibrator`. 



**Args:**
 
 - <b>`num_categories`</b>:  The number of known categories. 
 - <b>`missing_input_value`</b>:  If provided, the calibrator will learn to map all  instances of this missing input value to a learned output value just  the same as it does for known categories. Note that `num_categories`  will be one greater to include this missing category. 
 - <b>`output_min`</b>:  Minimum output value. If `None`, the minimum output value will  be unbounded. 
 - <b>`output_max`</b>:  Maximum output value. If `None`, the maximum output value will  be unbounded. 
 - <b>`monotonicity_pairs`</b>:  List of pairs of indices `(i,j)` indicating that the  calibrator output for index `j` should be greater than or equal to that  of index `i`. 
 - <b>`kernel_init`</b>:  Initialization scheme to use for the kernel. 



**Raises:**
 
 - <b>`ValueError`</b>:  If `monotonicity_pairs` is cyclic. 
 - <b>`ValueError`</b>:  If `kernel_init` is invalid. 




---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/categorical_calibrator/constrain#L140"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constrain`

```python
constrain() → None
```

Projects kernel into desired constraints. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/categorical_calibrator.py#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Tensor) → Tensor
```

Calibrates categorical inputs through a learned mapping. 



**Args:**
 
 - <b>`x`</b>:  The input tensor of category indices of shape `(batch_size, 1)`. 



**Returns:**
 torch.Tensor of shape `(batch_size, 1)` containing calibrated input values. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/categorical_calibrator/keypoints_inputs#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keypoints_inputs`

```python
keypoints_inputs() → Tensor
```

Returns a tensor of keypoint inputs (category indices). 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/categorical_calibrator/keypoints_outputs#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keypoints_outputs`

```python
keypoints_outputs() → Tensor
```

Returns a tensor of keypoint outputs. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
