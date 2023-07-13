<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_linear.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tests.layers.test_linear`
Tests for Linear module. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_linear.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_initialization`

```python
test_initialization(
    input_dim,
    monotonicities,
    use_bias,
    weighted_average
) → None
```

Tests that Linear initialization works properly 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_linear.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_forward`

```python
test_forward(kernel_data, bias_data, inputs, expected_outputs) → None
```

Tests that forward properly combined inputs. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_linear.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_no_constraints`

```python
test_constrain_no_constraints(monotonicities, kernel_data, bias_data) → None
```

Tests that constrain does nothing when there are no constraints. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_linear.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_monotonicities`

```python
test_constrain_monotonicities(
    monotonicities,
    kernel_data,
    expected_projected_kernel_data
) → None
```

Tests that constrain properly projects kernel according to monotonicies. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_linear.py#L145"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_weighted_average`

```python
test_constrain_weighted_average(
    kernel_data,
    expected_projected_kernel_data
) → None
```

Tests that constrain properly projects kernel to be a weighted average. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_linear.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_training`

```python
test_training() → None
```

Tests that the `Linear` module can learn f(x_1,x_2) = 2x_1 + 3x_2 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
