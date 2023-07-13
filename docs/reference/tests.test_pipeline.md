<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `tests.test_pipeline`
Tests for Pipeline. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_init`

```python
test_init(
    fixture_feature_names,
    fixture_target,
    target_type,
    expected_primary_metric
)
```

Tests pipeline initialization for a classification target. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_init_with_categories`

```python
test_init_with_categories(
    fixture_feature_names,
    fixture_target,
    fixture_categories_strs,
    fixture_categories_ints
)
```

Tests pipeline initialization with specified categories. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_init_from_config`

```python
test_init_from_config(
    fixture_target,
    fixture_feature_configs,
    target_type,
    metric,
    shuffle_data,
    drop_empty_percentage,
    dataset_split
)
```

Tests pipeline initialization from a `PipelineConfig` instance. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_prepare`

```python
test_prepare(
    fixture_data,
    fixture_feature_names,
    fixture_target,
    fixture_categories_strs
)
```

Tests the pipeline prepare function. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_train_calibrated_linear_model`

```python
test_train_calibrated_linear_model(
    fixture_data,
    fixture_feature_names,
    fixture_target,
    target_type
)
```

Tests pipeline training for calibrated linear regression model. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_pipeline_save_load`

```python
test_pipeline_save_load(
    fixture_data,
    fixture_feature_names,
    fixture_target,
    tmp_path
)
```

Tests that an instance of `Pipeline` can be successfully saved and loaded. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline/test_publish#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_publish`

```python
test_publish(post_pipeline, fixture_feature_names, fixture_target)
```

Tests that a pipeline can be published to the API. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline/test_analysis#L236"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_analysis`

```python
test_analysis(
    get_api_key,
    post_pipeline,
    post_pipeline_config,
    post_pipeline_feature_configs,
    post_trained_model_analysis,
    upload_model,
    fixture_data,
    fixture_feature_names,
    fixture_target
)
```

Tests that pipeline analysis works as expected. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline/test_run_inference#L290"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_run_inference`

```python
test_run_inference(
    get_api_key,
    post_inference,
    fixture_data,
    fixture_feature_names,
    fixture_target
)
```

Tests that a pipeline can run inference on a dataset. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/tests/test_pipeline/test_await_inference_results#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `test_await_inference_results`

```python
test_await_inference_results(
    get_inference_status,
    get_inference_results,
    fixture_data,
    fixture_feature_names,
    fixture_target
)
```

Tests that a pipeline can await inference results. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
