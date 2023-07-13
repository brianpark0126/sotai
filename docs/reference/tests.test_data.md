<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tests.test_data`
Tests for data utilities. 

**Global Variables**
---------------
- **MISSING_CATEGORY_VALUE**
- **MISSING_NUMERICAL_VALUE**

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data/fixture_categorical_data#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_categorical_data`

```python
fixture_categorical_data()
```

Returns a mapping for the header and data for categorical data. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data/fixture_numerical_data#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_numerical_data`

```python
fixture_numerical_data()
```

Returns a mapping for the header and data for numerical data. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data/fixture_target_data#L34"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_target_data`

```python
fixture_target_data()
```

Returns a mapping for the header and data for target data. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data/fixture_unknown_data#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_unknown_data`

```python
fixture_unknown_data()
```

Returns a mapping for the header and data for unknown data. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data/fixture_data#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_data`

```python
fixture_data(numerical_data, categorical_data, target_data)
```

Returns a dataframe containing the categorical, numerical, and target data. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data/fixture_categories#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_categories`

```python
fixture_categories(categorical_data)
```

Returns the categories for the categorical data. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data/fixture_missing_category#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_missing_category`

```python
fixture_missing_category(categorical_data)
```

Returns a missing category for the categorical data. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data/fixture_missing_input_value#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_missing_input_value`

```python
fixture_missing_input_value()
```

Returns a missing input value for the categorical data. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data/fixture_features#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_features`

```python
fixture_features(
    numerical_data,
    categorical_data,
    categories,
    missing_input_value
)
```

Returns a list of features. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data/fixture_feature_configs#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_feature_configs`

```python
fixture_feature_configs(numerical_data, categorical_data, categories)
```

Returns a list of feature configs. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data.py#L106"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_replace_missing_values`

```python
test_replace_missing_values(data, feature_configs)
```

Tests that missing values are properly replaced. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_determine_feature_types`

```python
test_determine_feature_types(data, unknown_data)
```

Tests the determination of feature types from data. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data.py#L124"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_initialization`

```python
test_initialization(from_filepath, tmp_path, data)
```

Tests that `CSVData` initialization works as expected. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data.py#L146"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_call`

```python
test_call(data, numerical_data, categorical_data, target_data)
```

Tests that the correct column data is returned when called. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data.py#L154"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_prepare`

```python
test_prepare(
    include_target,
    inplace,
    data,
    features,
    target_data,
    categorical_data,
    categories,
    missing_category,
    missing_input_value
)
```

Tests that the data is prepared as expected. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_data.py#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_batch`

```python
test_batch(
    include_target,
    batch_size,
    expected_example_batches,
    expected_target_batches,
    data,
    features,
    target_data
)
```

Tests that batches of data are properly generated. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
