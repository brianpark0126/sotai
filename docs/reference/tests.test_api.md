<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_api.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tests.test_api`
Tests for api. 

**Global Variables**
---------------
- **SOTAI_API_ENDPOINT**
- **SOTAI_BASE_URL**

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_api/test_post_pipeline#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_post_pipeline`

```python
test_post_pipeline(mock_get_api_key, mock_post, fixture_pipeline)
```

Tests that a pipeline is posted correctly. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_api/test_post_pipeline_config#L42"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_post_pipeline_config`

```python
test_post_pipeline_config(mock_get_api_key, mock_post, fixture_pipeline_config)
```

Tests that a pipeline config is posted correctly. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_api/test_post_feature_configs#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_post_feature_configs`

```python
test_post_feature_configs(
    mock_get_api_key,
    mock_post,
    fixture_pipeline_config,
    fixture_categories_strs,
    fixture_categories_ints
)
```

Tests that feature configs are posted correctly. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_api/test_post_trained_model_analysis#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_post_trained_model_analysis`

```python
test_post_trained_model_analysis(
    mock_get_api_key,
    mock_post,
    fixture_trained_model
)
```

Tests that a trained model is posted correctly. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_api/test_post_trained_model#L174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_post_trained_model`

```python
test_post_trained_model(
    mock_get_api_key,
    mock_post,
    mock_open_data,
    mock_tarfile_open
)
```

Tests that feature configs are posted correctly. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_api/test_post_inferencel#L199"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_post_inferencel`

```python
test_post_inferencel(mock_get_api_key, mock_post, mock_open_data)
```

Tests that feature configs are posted correctly. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_api/test_get_inference_status#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_get_inference_status`

```python
test_get_inference_status(mock_get_api_key, mock_get)
```

Tests that inference config retrieval is handled correctly. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_api/test_get_inference_result#L244"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_get_inference_result`

```python
test_get_inference_result(mock_get_api_key, mock_get, mock_urlretrieve)
```

Tests that inference file retrieval is handled correctly. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
