<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/numerical_calibrator.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.layers.numerical_calibrator`
Numerical calibration module. 

PyTorch implementation of the numerical calibration module. This module takes in a single-dimensional input and transforms it using piece-wise linear functions that satisfy desired bounds and monotonicity constraints. 



---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/numerical_calibrator.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NumericalCalibrator`
A numerical calibrator. 

This module takes an input of shape `(batch_size, 1)` and calibrates it using a piece-wise linear function that conforms to any provided constraints. The output will have the same shape as the input. 



**Attributes:**
 
    - All `__init__` arguments. 
 - <b>`kernel`</b>:  `torch.nn.Parameter` that stores the piece-wise linear function weights. 
 - <b>`missing_output`</b>:  `torch.nn.Parameter` that stores the output learned for any  missing inputs. Only available if `missing_input_value` is provided. 



**Example:**
 

```python
inputs = torch.tensor(...)  # shape: (batch_size, 1)
calibrator = NumericalCalibrator(
    input_keypoints=np.linspace(1., 5., num=5),
    output_min=0.0,
    output_max=1.0,
    monotonicity=Monotonicity.INCREASING,
    kernel_init=NumericalCalibratorInit.EQUAL_HEIGHTS,
)
outputs = calibrator(inputs)
``` 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/numerical_calibrator.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_keypoints: ndarray,
    missing_input_value: Optional[float] = None,
    output_min: Optional[float] = None,
    output_max: Optional[float] = None,
    monotonicity: Monotonicity = 'none',
    kernel_init: NumericalCalibratorInit = 'equal_heights',
    projection_iterations: int = 8
) → None
```

Initializes an instance of `NumericalCalibrator`. 



**Args:**
 
 - <b>`input_keypoints`</b>:  Ordered list of float-valued keypoints for the underlying  piece-wise linear function. 
 - <b>`missing_input_value`</b>:  If provided, the calibrator will learn to map all  instances of this missing input value to a learned output value. 
 - <b>`output_min`</b>:  Minimum output value. If `None`, the minimum output value will  be unbounded. 
 - <b>`output_max`</b>:  Maximum output value. If `None`, the maximum output value will  be unbounded. 
 - <b>`monotonicity`</b>:  Monotonicity constraint for the underlying piece-wise linear  function. 
 - <b>`kernel_init`</b>:  Initialization scheme to use for the kernel. 
 - <b>`projectionion_iterations`</b>:  Number of times to run Dykstra's projection  algorithm when applying constraints. 



**Raises:**
 
 - <b>`ValueError`</b>:  If `kernel_init` is invalid. 




---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/numerical_calibrator/constrain#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constrain`

```python
constrain() → None
```

Jointly projects kernel into desired constraints. 

Uses Dykstra's alternating projection algorithm to jointly project onto all given constraints. This algorithm projects with respect to the L2 norm, but it approached the norm from the "wrong" side. To ensure that all constraints are strictly met, we do final approximate projections that project strictly into the feasible space, but this is not an exact projection with respect to the L2 norm. Enough iterations make the impact of this approximation negligible. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/numerical_calibrator.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Tensor) → Tensor
```

Calibrates numerical inputs through piece-wise linear interpolation. 



**Args:**
 
 - <b>`x`</b>:  The input tensor of shape `(batch_size, 1)`. 



**Returns:**
 torch.Tensor of shape `(batch_size, 1)` containing calibrated input values. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/numerical_calibrator/keypoints_inputs#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keypoints_inputs`

```python
keypoints_inputs() → Tensor
```

Returns tensor of keypoint inputs. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/numerical_calibrator/keypoints_outputs#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `keypoints_outputs`

```python
keypoints_outputs() → Tensor
```

Returns tensor of keypoint outputs. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
