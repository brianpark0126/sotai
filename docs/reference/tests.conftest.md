<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/conftest.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tests.conftest`
Fixtures to help with testing. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/conftest/fixture_target#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_target`

```python
fixture_target()
```

Returns a test target. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/conftest/fixture_categories_strs#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_categories_strs`

```python
fixture_categories_strs()
```

Returns a list of test string categories. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/conftest/fixture_categories_ints#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_categories_ints`

```python
fixture_categories_ints()
```

Returns of a list of test integer categories. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/conftest/fixture_data#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_data`

```python
fixture_data(fixture_categories_strs, fixture_categories_ints, fixture_target)
```

Returns a test dataset. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/conftest/fixture_feature_names#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_feature_names`

```python
fixture_feature_names(fixture_data, fixture_target)
```

Returns a list of test feature names. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/conftest/fixture_feature_configs#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_feature_configs`

```python
fixture_feature_configs(
    fixture_categories_strs,
    fixture_categories_ints
) → Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]]
```

Returns a list of test features. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/conftest/fixture_pipeline#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_pipeline`

```python
fixture_pipeline(
    fixture_target,
    fixture_feature_names,
    fixture_categories_strs,
    fixture_categories_ints
) → Pipeline
```

Returns a list of test features. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/conftest/fixture_pipeline_config#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_pipeline_config`

```python
fixture_pipeline_config(
    fixture_target,
    fixture_categories_strs,
    fixture_categories_ints
) → PipelineConfig
```

Returns a pipeline config that can be used for testing 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/conftest/fixture_trained_model#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fixture_trained_model`

```python
fixture_trained_model(fixture_pipeline_config) → TrainedModel
```

Returns a trained model that can be used for testing. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
