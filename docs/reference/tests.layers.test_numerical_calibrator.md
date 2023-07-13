<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tests.layers.test_numerical_calibrator`
Tests for NumericalCalibrator module. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_initialization`

```python
test_initialization(
    input_keypoints,
    missing_input_value,
    output_min,
    output_max,
    monotonicity,
    kernel_init,
    projection_iterations,
    expected_kernel
)
```

Tests that NumericalCalibrator class initialization works properly. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_forward`

```python
test_forward(
    input_keypoints,
    kernel_init,
    kernel_data,
    inputs,
    expected_outputs
)
```

Tests that forward properly calibrated inputs. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_no_constraints`

```python
test_constrain_no_constraints()
```

Tests that constrain does nothing when there are no constraints. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_only_output_min`

```python
test_constrain_only_output_min(output_min, kernel_data)
```

Tests that constrain properly projects kernel into output_min constraint. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L198"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_only_output_max`

```python
test_constrain_only_output_max(output_max, kernel_data)
```

Tests that constrain properly projects kernel into output_max constraint. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L215"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_bounds`

```python
test_constrain_bounds(output_min, output_max, kernel_data)
```

Tests that constrain properly projects kernel into output bounds. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_increasing_monotonicity`

```python
test_constrain_increasing_monotonicity(kernel_data)
```

Tests that contrain properly projects kernel to be increasingly monotonic. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L258"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_decreasing_monotonicity`

```python
test_constrain_decreasing_monotonicity(kernel_data)
```

Tests that contrain properly projects kernel to be decreasingly monotonic. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L276"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_output_min_monotonicity`

```python
test_constrain_output_min_monotonicity(output_min, monotonicity, kernel_data)
```

Tests contraining output min with monotonicity constraints. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L319"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_output_max_with_monotonicity`

```python
test_constrain_output_max_with_monotonicity(
    output_max,
    monotonicity,
    kernel_data
)
```

Tests contraining output max with monotonicity constraints. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L362"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_bounds_with_monotonicity`

```python
test_constrain_bounds_with_monotonicity(
    output_min,
    output_max,
    monotonicity,
    kernel_data
)
```

Tests constraining output bounds with monotonicity constraints. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L403"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_keypoints_inputs`

```python
test_keypoints_inputs(input_keypoints)
```

Tests that the correct keypoint inputs are returned. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L413"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_keypoints_outputs`

```python
test_keypoints_outputs(num_keypoints, kernel_data, expected_keypoints_outputs)
```

Tests that the correct keypoint outputs are returned. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L435"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_project_monotonic_bounds`

```python
test_project_monotonic_bounds(
    input_keypoints,
    output_min,
    output_max,
    monotonicity,
    kernel_data,
    expected_projected_kernel_data
)
```

Tests that kernel is properly projected into bounds with monotonicity. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L563"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_approximately_project_bounds_only`

```python
test_approximately_project_bounds_only(
    output_min,
    output_max,
    kernel_data,
    expected_projected_kernel_data
)
```

Tests that bounds are properly projected when monotonicity is NONE. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L628"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_project_monotonicity`

```python
test_project_monotonicity(monotonicity, heights, expected_projected_heights)
```

Tests that monotonicity is properly projected 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L658"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_squeeze_by_scaling`

```python
test_squeeze_by_scaling(
    monotonicity,
    output_min,
    output_max,
    kernel_data,
    expected_projected_kernel_data
)
```

Tests that kernel is scaled into bound constraints properly. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_numerical_calibrator.py#L724"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_training`

```python
test_training()
```

Tests that the `NumericalCalibrator` module can learn f(x) = |x|. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
