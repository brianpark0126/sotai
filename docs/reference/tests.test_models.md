<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_models.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tests.test_models`
Tests for models. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_models.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_initialization`

```python
test_initialization(
    features,
    output_min,
    output_max,
    output_calibration_num_keypoints,
    expected_linear_monotonicities,
    expected_output_calibrator_monotonicity
)
```

Tests that `CalibratedLinear` initialization works. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_models.py#L137"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_forward`

```python
test_forward(
    output_min,
    output_max,
    calibrator_kernel_datas,
    linear_kernel_data,
    output_calibrator_kernel_data,
    inputs,
    expected_outputs
)
```

Tests that forward returns expected result. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_models.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_constrain`

```python
test_constrain()
```

Tests that constrain properly constrains all layers. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_models.py#L250"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_training`

```python
test_training()
```

Tests `CalibratedLinear` training on data from f(x) = 0.7|x_1| + 0.3x_2. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
