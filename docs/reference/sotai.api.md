<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.api`
This module contains the API functions for interacting with the SOTAI API. 

**Global Variables**
---------------
- **SOTAI_API_ENDPOINT**
- **SOTAI_API_TIMEOUT**
- **SOTAI_BASE_URL**

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `set_api_key`

```python
set_api_key(api_key: str)
```

Set the SOTAI API key in the environment variables. 



**Args:**
 
 - <b>`api_key`</b>:  The API key to set. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_api_key`

```python
get_api_key() → str
```

Returns the SOTAI API key from the environment variables. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_auth_headers`

```python
get_auth_headers() → Dict[str, str]
```

Returns the authentication headers for a pipeline. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_pipeline`

```python
post_pipeline(pipeline) → Tuple[APIStatus, Optional[str]]
```

Create a new pipeline on the SOTAI API. 



**Args:**
 
 - <b>`pipeline`</b>:  The pipeline to create. 



**Returns:**
 A tuple containing the status of the API call and the UUID of the created pipeline. If unsuccessful, the UUID will be `None`. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L76"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_pipeline_config`

```python
post_pipeline_config(
    pipeline_uuid: str,
    pipeline_config: PipelineConfig
) → Tuple[APIStatus, Optional[str]]
```

Create a new pipeline config on the SOTAI API. 



**Args:**
 
 - <b>`pipeline_uuid`</b>:  The pipeline uuid to create the pipeline config for. 
 - <b>`pipeline_config `</b>:  The pipeline config to create. 



**Returns:**
 A tuple containing the status of the API call and the UUID of the created pipeline. If unsuccessful, the UUID will be `None`. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_pipeline_feature_configs`

```python
post_pipeline_feature_configs(
    pipeline_config_uuid: str,
    feature_configs: Dict[str, Union[CategoricalFeatureConfig, NumericalFeatureConfig]]
) → <enum 'APIStatus'>
```

Create a new pipeline feature configs on the SOTAI API. 



**Args:**
 
 - <b>`pipeline_config_uuid`</b>:  The pipeline config uuid to create the pipeline  feature configs for. 
 - <b>`feature_configs`</b>:  The feature configs to create. 



**Returns:**
 The status of the API call. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_trained_model_analysis`

```python
post_trained_model_analysis(
    pipeline_config_uuid: str,
    trained_model: TrainedModelMetadata
) → Tuple[APIStatus, Optional[Dict[str, str]]]
```

Create a new trained model analysis on the SOTAI API. 



**Args:**
 
 - <b>`pipeline_config_uuid`</b>:  The pipeline config uuid to create the trained model  analysis for. 
 - <b>`trained_model`</b>:  The trained model to create. 



**Returns:**
 A tuple containing the status of the API call and a dict containing the UUIDs of the resources created as well as a link that can be used to view the trained model analysis. If unsuccessful, the UUID will be `None`. 

Keys: 
        - `trained_model_metadata_uuid`: The UUID of the trained model. 
        - `model_config_uuid`: The UUID of the model configuration. 
        - `pipeline_config_uuid`: The UUID of the pipeline configuration. 
        - `analysis_url`: The URL of the trained model analysis. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_trained_model`

```python
post_trained_model(
    trained_model_path: str,
    trained_model_uuid: str
) → <enum 'APIStatus'>
```

Create a new trained model on the SOTAI API. 



**Args:**
 
 - <b>`trained_model_path`</b>:  The path to the trained model file to post. 
 - <b>`trained_model_uuid`</b>:  The UUID of the trained model. 



**Returns:**
 The status of the API call. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L294"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
 A tuple containing the status of the API call and the UUID of the created inference job. If unsuccessful, the UUID will be `None`. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L325"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_inference_status`

```python
get_inference_status(
    inference_uuid: str
) → Tuple[APIStatus, Optional[InferenceConfigStatus]]
```

Get an inference from the SOTAI API. 



**Args:**
 
 - <b>`inference_uuid`</b>:  The UUID of the inference to get. 



**Returns:**
 A tuple containing the status of the API call and the status of the inference job if the API call is successful. If unsuccessful, the UUID will be `None`. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L351"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_inference_results`

```python
get_inference_results(
    inference_uuid: str,
    download_folder: str
) → <enum 'APIStatus'>
```

Get an inference from the SOTAI API. 



**Args:**
 
 - <b>`inference_uuid`</b>:  The UUID of the inference results to get. 



**Returns:**
 The status of the API call. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L378"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_dataset`

```python
post_dataset(
    data_filepath: str,
    columns: List[str],
    categorical_columns: List[str],
    pipeline_uuid: str
) → Tuple[APIStatus, Optional[str]]
```

Upload a dataset to th the SOTAI API. 



**Args:**
 
 - <b>`data_filepath`</b>:  The path to the data file to push to the API. 
 - <b>`columns`</b>:  The columns of the dataset. 
 - <b>`categorical_columns`</b>:  The categorical columns of the dataset. 
 - <b>`pipeline_uuid`</b>:  The pipeline uuid for which to upload the dataset. 



**Returns:**
 A tuple containing the status of the API call and the UUID of the created dataset. If unsuccessful, the UUID will be `None`. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `post_hypertune_job`

```python
post_hypertune_job(
    hypertune_config: HypertuneConfig,
    pipeline_config: PipelineConfig,
    model_config: Type[_BaseModelConfig],
    dataset_uuid: str,
    next_model_id: int
)
```

Upload a dataset to th the SOTAI API. 



**Args:**
 
 - <b>`hypertune_config`</b>:  The hypertune config to create the hypertune job for. 
 - <b>`pipeline_config`</b>:  The pipeline config to create the hypertune job for. 
 - <b>`dataset_uuid`</b>:  The dataset uuid to create the hypertune job for. 



**Returns:**
 A tuple containing the status of the API call and an array of the UUIDs of the created trained models. If unsuccessful, the UUIDs will be `None`. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L523"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_pipeline`

```python
get_pipeline(
    pipeline_uuid: str
) → Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str]]
```

Get a pipeline from the SOTAI API. 



**Args:**
 
 - <b>`pipeline_uuid`</b>:  The UUID of the pipeline to get. 



**Returns:**
 A tuple containing the metadata for the pipeline, the id for the most recent config of the pipeline, the pipeline configs for the pipeline, and the UUIDs of the trainedmodels for the pipeline. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L566"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_trained_model_uuids`

```python
get_trained_model_uuids(pipeline_uuid: str) → List[str]
```

Get the UUIDs of the trained models for a pipeline from the SOTAI API. 



**Args:**
 
 - <b>`pipeline_uuid`</b>:  The UUID of the pipeline to get the trained models for. 



**Returns:**
 The UUIDs of the trained models for the pipeline. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L584"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_trained_model_metadata`

```python
get_trained_model_metadata(trained_model_uuid: str) → TrainedModelMetadata
```

Get the metadata for a TrainedModelfrom the SOTAI API. 



**Args:**
 
 - <b>`trained_model_uuid`</b>:  The UUID of the trained model to get. 



**Returns:**
 The metadata for the trained model. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/api.py#L675"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `download_trained_model`

```python
download_trained_model(trained_model_uuid: str)
```

Download a trained model from the SOTAI API to a local tmp directory. 



**Args:**
 
 - <b>`trained_model_uuid`</b>:  The UUID of the trained model to download. 



**Returns:**
 The path to the downloaded model. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
