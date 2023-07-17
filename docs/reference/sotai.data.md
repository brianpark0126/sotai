<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/data.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.data`
Useful classes and functions for handling data for calibrated modeling. 

**Global Variables**
---------------
- **MISSING_CATEGORY_VALUE**
- **MISSING_NUMERICAL_VALUE**

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/data.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `replace_missing_values`

```python
replace_missing_values(
    data: DataFrame,
    feature_configs: Dict[str, Union[CategoricalFeatureConfig, NumericalFeatureConfig]]
) → DataFrame
```

Replaces empty values or unspecified categories with a constant value. 



**Args:**
 
    - data: The dataset in which to replace missing values. 
    - feature_configs: A dictionary mapping feature names to feature configurations. 



**Returns:**
  The dataset with missing values replaced. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/data.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `determine_feature_types`

```python
determine_feature_types(data: DataFrame) → Dict[str, FeatureType]
```

Returns a dictionary mapping feature name to type for the given data. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/data.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CSVData`
Class for handling CSV data for calibrated modeling. 



**Attributes:**
 
    - All `__init__` arguments. 
 - <b>`data`</b>:  A pandas `DataFrame` containing the loaded CSV data. 
 - <b>`headers`</b>:  The list of headers available from the loaded data. 
 - <b>`num_examples`</b>:  The number of examples in the dataset. 
 - <b>`prepared_data`</b>:  The prepared data. This will be `None` if `prepare(...)` has not  been called. 



**Example:**
 ```python
csv_data = CSVData("path/to/data.csv")
feature_configs = [
    NumericalFeatureConfig(
         feature_name="numerical_feature"
         data=csv_data("numerical_feature")  # must match feature column header
    ),
    CategoricalFeatureConfig(
         feature_name="categorical_feature"
         categories=np.unique(csv_data("categorical_feature"))  # uses all categories
    ),
]
csv_data.prepare(feature_configs, "target", ...)  # must match target column header
for examples, labels in csv_data.batch(64):
    training_step(...)
``` 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/data.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(dataset: Union[str, DataFrame]) → None
```

Initializes an instance of `CSVData`. 

Loads a CSV file if filepath is provided. Otherwise it will use the provided DataFrame. 



**Args:**
 
 - <b>`dataset`</b>:  Either a string filepath pointing to the CSV data that should be  loaded or a `pd.DataFrame` containing the data that should be used.  The CSV file or DataFrame must have a header. 




---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/data.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `batch`

```python
batch(
    batch_size: int
) → Generator[Union[Tuple[Tensor, Tensor], Tensor], NoneType, NoneType]
```

A generator that yields a tensor with `batch_size` examples. 



**Args:**
 
 - <b>`batch_size`</b>:  The size of each batch returns during each iteration. 



**Yields:**
 
 - <b>`If prepared with a target column`</b>:  a tuple (examples, targets) of  `torch.Tensor` of shape `(batch_size, num_features)` and  `(batch_size, 1)`, repsectively. 
 - <b>`If prepared without a target column`</b>:  a `torch.Tensor` of shape  `(batch_size, num_features)`. 



**Raises:**
 
 - <b>`ValueError`</b>:  If `prepare(...)` is not called first. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/data.py#L142"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare`

```python
prepare(
    features: List[Union[CategoricalFeature, NumericalFeature]],
    target_header: Optional[str],
    inplace: bool = True
) → None
```

Prepares the data for calibrated modeling. 



**Args:**
 
 - <b>`feature_configs`</b>:  Feature configs that specify how to prepare the data. 
 - <b>`target_header`</b>:  The header for the target column. If `None`, it will be  assumed that there is no target column present (e.g. for inference) 
 - <b>`inplace`</b>:  If True, original `data` attribute will be updated. If False, a  copy of the original data will be prepared and the original will be  preserved. 



**Raises:**
 
 - <b>`ValueError`</b>:  If a feature in `feature_configs` is not in the dataset. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
