<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tests.utils`
Testing Utilities. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/utils.py#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_calibrated_module`

```python
train_calibrated_module(
    calibrated_module: Module,
    examples: Tensor,
    labels: Tensor,
    loss_fn: Module,
    optimizer: Optimizer,
    epochs: int,
    batch_size: int
)
```

Trains a calibrated module for testing purposes. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/utils.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `construct_trained_model`

```python
construct_trained_model(
    target_type: TargetType,
    data: DataFrame,
    feature_configs: Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]]
)
```

Returns a `TrainedModel` instance. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/utils.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MockResponse`
Mock response class for testing. 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/utils.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(json_data, status_code=200)
```

Mock response for testing. 




---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/utils.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `json`

```python
json()
```

Return json data. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
