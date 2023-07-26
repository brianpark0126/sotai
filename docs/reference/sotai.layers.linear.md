<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/linear.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.layers.linear`
Linear module for use in calibrated modeling. 

PyTorch implementation of the calibrated linear module. This module takes in a single-dimensional input and transforms it using a linear transformation and optionally a bias term. This module supports monotonicity constraints. 



---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/linear.py#L14"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Linear`
A linear module. 

This module takes an input of shape `(batch_size, input_dim)` and applied a linear transformation. The output will have the same shape as the input. 



**Attributes:**
 
  - All `__init__` arguments. 
 - <b>`kernel`</b>:  `torch.nn.Parameter` that stores the linear combination weighting. 
 - <b>`bias`</b>:  `torch.nn.Parameter` that stores the bias term. Only available is `use_bias`  is true. 



**Example:**
 ```python
input_dim = 3
inputs = torch.tensor(...)  # shape: (batch_size, input_dim)
linear = Linear(
  input_dim,
  monotonicities=[
     Monotonicity.NONE,
     Monotonicity.INCREASING,
     Monotonicity.DECREASING
  ],
  use_bias=False,
  weighted_average=True,
)
outputs = linear(inputs)
``` 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/linear.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_dim,
    monotonicities: Optional[List[Monotonicity]] = None,
    use_bias: bool = True,
    weighted_average: bool = False
) → None
```

Initializes an instance of `Linear`. 



**Args:**
 
 - <b>`input_dim`</b>:  The number of inputs that will be combined. 
 - <b>`monotonicities`</b>:  If provided, specifies the monotonicity of each input  dimension. 
 - <b>`use_bias`</b>:  Whether to use a bias term for the linear combination. 
 - <b>`weighted_average`</b>:  Whether to make the output a weighted average i.e. all  coefficients are positive and add up to a total of 1.0. No bias term will  be used, and `use_bias` will be set to false regardless of the original  value. `monotonicities` will also be set to increasing for all input  dimensions to ensure that all coefficients are positive. 



**Raises:**
 
 - <b>`ValueError`</b>:  If monotonicities does not have length input_dim (if provided). 




---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/linear/assert_constraints#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `assert_constraints`

```python
assert_constraints(eps=1e-06) → List[str]
```

Asserts that layer satisfies specified constraints. 

This checks that decreasing monotonicity corresponds to negative weights, increasing monotonicity corresponds to positive weights, and weights sum to 1 for weighted_average=True. 



**Args:**
 
 - <b>`eps`</b>:  the margin of error allowed 



**Returns:**
 A list of messages describing violated constraints. If no constraints violated, the list will be empty. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/linear/constrain#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constrain`

```python
constrain() → None
```

Projects kernel into desired constraints. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/layers/linear.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Tensor) → Tensor
```

Transforms inputs using a linear combination. 



**Args:**
 
 - <b>`x`</b>:  The input tensor of shape `(batch_size, input_dim)`. 



**Returns:**
 torch.Tensor of shape `(batch_size, 1)` containing transformed input values. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
