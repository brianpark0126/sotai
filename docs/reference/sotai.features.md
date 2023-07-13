<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/features.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.features`
Feature objects for use in models. 

To construct a calibrated model, create the calibrated model configuration and pass it in to the corresponding calibrated model constructor. 



**Example:**
 ```python
feature_configs = [...]
linear_config = CalibratedLinearConfig(feature_configs, ...)
linear_model = CalibratedLinear(linear_config)
``` 



---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/features.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NumericalFeature`
Feature configuration for numerical features. 



**Attributes:**
 
    - All `__init__` arguments. 
 - <b>`feature_type`</b>:  The type of this feature. Always `FeatureType.NUMERICAL`. 
 - <b>`input_keypoints`</b>:  The input keypoints used for this feature's calibrator. These  keypoints will be initialized using the given `data` under the desired  `input_keypoints_init` scheme. 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/features.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    feature_name: str,
    data: ndarray,
    num_keypoints: int = 10,
    input_keypoints_init: InputKeypointsInit = 'quantiles',
    missing_input_value: Optional[float] = None,
    monotonicity: Monotonicity = 'none'
) → None
```

Initializes a `NumericalFeatureConfig` instance. 



**Args:**
 
 - <b>`feature_name`</b>:  The name of the feature. This should match the header for the  column in the dataset representing this feature. 
 - <b>`data`</b>:  Numpy array of float-valued data used for calculating keypoint inputs  and initializing keypoint outputs. 
 - <b>`num_keypoints`</b>:  The number of keypoints used by the underlying piece-wise  linear function of a NumericalCalibrator. There will be  `num_keypoints - 1` total segments. 
 - <b>`input_keypoints_init`</b>:  The scheme to use for initializing the input  keypoints. See `InputKeypointsInit` for more details. 
 - <b>`missing_input_value`</b>:  If provided, this feature's calibrator will learn to  map all instances of this missing input value to a learned output value. 
 - <b>`monotonicity`</b>:  Monotonicity constraint for this feature, if any. 



**Raises:**
 
 - <b>`ValueError`</b>:  If `data` contains NaN values. 
 - <b>`ValueError`</b>:  If `input_keypoints_init` is invalid. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/features.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CategoricalFeature`
Feature configuration for categorical features. 



**Attributes:**
 
    - All `__init__` arguments. 
 - <b>`feature_type`</b>:  The type of this feature. Always `FeatureType.CATEGORICAL`. 
 - <b>`category_indices`</b>:  A dictionary mapping string categories to their index. 
 - <b>`monotonicity_index_pairs`</b>:  A conversion of `monotonicity_pairs` from string  categories to category indices. Only available if `monotonicity_pairs` are  provided. 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/features.py#L125"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    feature_name: str,
    categories: Union[List[int], List[str]],
    missing_input_value: Optional[float] = None,
    monotonicity_pairs: Optional[List[Tuple[str, str]]] = None
) → None
```

Initializes a `CategoricalFeatureConfig` instance. 



**Args:**
 
 - <b>`feature_name`</b>:  The name of the feature. This should match the header for the  column in the dataset representing this feature. 
 - <b>`categories`</b>:  The categories that should be used for this feature. Any  categories not contained will be considered missing or unknown. If you  expect to have such missing categories, make sure to 
 - <b>`missing_input_value`</b>:  If provided, this feature's calibrator will learn to  map all instances of this missing input value to a learned output value. 
 - <b>`monotonicity_pairs`</b>:  List of pairs of categories `(category_a, category_b)`  indicating that the calibrator output for `category_b` should be greater  than or equal to that of `category_a`. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
