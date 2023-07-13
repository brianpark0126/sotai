<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.api`
This module contains the API functions for interacting with the SOTAI API. 

**Global Variables**
---------------
- **SOTAI_API_ENDPOINT**
- **SOTAI_BASE_URL**
- **SOTAI_API_TIMEOUT**

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_api_key`

```python
set_api_key(api_key: str)
```

Set the SOTAI API key in the environment variables. 



**Args:**
 
 - <b>`api_key`</b>:  The API key to set. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_api_key`

```python
get_api_key() → str
```

Get the SOTAI API key from the environment variables. 



**Returns:**
  The API key retrieved from the environment variable. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_auth_headers`

```python
get_auth_headers() → Dict[str, str]
```

Get the authentication headers for a pipeline. 



**Returns:**
  The authentication headers. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L50"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_pipeline`

```python
post_pipeline(pipeline) → Tuple[APIStatus, Optional[str]]
```

Create a new pipeline on the SOTAI API . 



**Args:**
 
 - <b>`pipeline`</b>:  The pipeline to create. 



**Returns:**
 A tuple containing the status of the API call and the UUID of the created pipeline. If unsuccessful, the UUID will be None. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_pipeline_config`

```python
post_pipeline_config(
    pipeline_uuid: str,
    pipeline_config: PipelineConfig
) → Tuple[APIStatus, Optional[str]]
```

Create a new pipeline config on the SOTAI API . 



**Args:**
 
 - <b>`pipeline_uuid`</b>:  The pipeline uuid to create the pipeline config for. 
 - <b>`pipeline_config `</b>:  The pipeline config to create. 



**Returns:**
 A tuple containing the status of the API call and the UUID of the created pipeline. If unsuccessful, the UUID will be None. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_pipeline_feature_configs`

```python
post_pipeline_feature_configs(
    pipeline_config_uuid: str,
    feature_configs: Dict[str, Union[CategoricalFeatureConfig, NumericalFeatureConfig]]
) → <enum 'APIStatus'>
```

Create a new pipeline feature configs on the SOTAI API . 



**Args:**
 
 - <b>`pipeline_config_uuid`</b>:  The pipeline config uuid to create the pipeline  feature configs for. 
 - <b>`feature_configs`</b>:  The feature configs to create. 



**Returns:**
 The status of the API call. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L165"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_trained_model_analysis`

```python
post_trained_model_analysis(
    pipeline_config_uuid: str,
    trained_model: TrainedModel
) → Tuple[APIStatus, Optional[Dict[str, str]]]
```

Create a new trained model analysis on the SOTAI API . 



**Args:**
 
 - <b>`pipeline_config_uuid`</b>:  The pipeline config uuid to create the trained model  analysis for. 
 - <b>`trained_model`</b>:  The trained model to create. 



**Returns:**
 A tuple containing the status of the API call and a dict containing the UUIDs of the resources created as well as a link that can be used to view the trained model analysis. If unsuccessful, the UUID will be None. 

Keys: 
        - `trainedModelMetadataUUID`: The UUID of the trained model. 
        - `modelConfigUUID`: The UUID of the model configuration. 
        - `pipelineConfigUUID`: The UUID of the pipeline configuration. 
        - `analysisURL`: The URL of the trained model analysis. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_trained_model`

```python
post_trained_model(
    trained_model_path: str,
    trained_model_uuid: str
) → <enum 'APIStatus'>
```

Create a new trained model on the SOTAI API . 



**Args:**
 
 - <b>`trained_model_path`</b>:  The path to the trained model file to post. 
 - <b>`trained_model_uuid`</b>:  The UUID of the trained model. 



**Returns:**
 The status of the API call. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_inference`

```python
post_inference(
    data_filepath: str,
    trained_model_uuid: str
) → Tuple[APIStatus, Optional[str]]
```

Create a new inference on the SOTAI API . 



**Args:**
 
 - <b>`data_filepath`</b>:  The path to the data file to create the inference for. 
 - <b>`trained_model_uuid`</b>:  The trained model uuid to create the inference for. 



**Returns:**
 A tuple containing the status of the API call and the UUID of the created inference job. If unsuccessful, the UUID will be None. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L323"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_inference_status`

```python
get_inference_status(
    inference_uuid: str
) → Tuple[APIStatus, Optional[InferenceConfigStatus]]
```

Get an inference from the SOTAI API . 



**Args:**
 
 - <b>`inference_uuid`</b>:  The UUID of the inference to get. 



**Returns:**
 A tuple containing the status of the API call and the status of the inference job if the API call is successful. If unsuccessful, the UUID will be None. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L349"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_inference_results`

```python
get_inference_results(
    inference_uuid: str,
    download_folder: str
) → <enum 'APIStatus'>
```

Get an inference from the SOTAI API . 



**Args:**
 
 - <b>`inference_uuid`</b>:  The UUID of the inference to get. 



**Returns:**
 The status of the API call. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
