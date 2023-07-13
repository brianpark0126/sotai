<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tests.layers.test_categorical_calibrator`
Tests for CategoricalCalibrator module. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L12"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_initialization`

```python
test_initialization(
    num_categories,
    missing_input_value,
    output_min,
    output_max,
    monotonicity_pairs,
    kernel_init
)
```

Tests that CategoricalCalibrator initialization works properly. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_forward`

```python
test_forward(missing_input_value, kernel_data, inputs, expected_outputs)
```

Tests that forward properly calibrated inputs. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_no_constraints`

```python
test_constrain_no_constraints()
```

Tests that constain does nothing when there are no costraints. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_only_output_min`

```python
test_constrain_only_output_min(output_min, kernel_data)
```

Tests that constrain properly projects kernel into output_min constraint. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_only_output_max`

```python
test_constrain_only_output_max(output_max, kernel_data)
```

Tests that constrain properly projects kernel into output_max constraint. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_bounds`

```python
test_constrain_bounds(output_min, output_max, kernel_data)
```

Tests that constrain properly projects kernel into output bounds. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L158"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_monotonicity_pairs`

```python
test_constrain_monotonicity_pairs(monotonicity_pairs, kernel_data)
```

Tests that contrain properly projects kernel to match monotonicity pairs. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L181"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_output_min_with_monotonicity_pairs`

```python
test_constrain_output_min_with_monotonicity_pairs(
    output_min,
    monotonicity_pairs,
    kernel_data
)
```

Tests constaining output min with monotonicity pairs. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_output_max_with_monotonicity_pairs`

```python
test_constrain_output_max_with_monotonicity_pairs(
    output_max,
    monotonicity_pairs,
    kernel_data
)
```

Tests constaining output max with monotonicity pairs. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L231"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain_bounds_with_monotonicity_pairs`

```python
test_constrain_bounds_with_monotonicity_pairs(
    output_min,
    output_max,
    monotonicity_pairs,
    kernel_data
)
```

Tests constaining bounds with monotonicity pairs. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L270"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_keypoints_inputs`

```python
test_keypoints_inputs(
    num_categories,
    missing_input_value,
    expected_keypoints_inputs
)
```

Tests that the correct keypoint inputs are returned. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_keypoints_outputs`

```python
test_keypoints_outputs(kernel_data, expected_keypoints_outputs)
```

Tests that the correct keypoint outputs are returned. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L305"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_approximately_project_monotonicity_pairs`

```python
test_approximately_project_monotonicity_pairs(
    monotonicity_pairs,
    kernel_data,
    expected_projected_kernel_data
)
```

Tests that kernel is properly projected to match monotonicity pairs. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/layers/test_categorical_calibrator.py#L340"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_training`

```python
test_training()
```

Tests that the `CategoricalCalibrator` module can learn a mapping. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
