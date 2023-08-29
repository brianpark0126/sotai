<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.pipeline`
A Pipeline for calibrated modeling. 

**Global Variables**
---------------
- **INFERENCE_POLLING_INTERVAL**
- **SOTAI_BASE_URL**


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Pipeline`
A pipeline for calibrated modeling. 

The pipeline defines the configuration for training a calibrated model. The pipeline itself defines the features, target, and target type to be used. When training a model, the data and configuration used will be versioned and stored in the pipeline. The pipeline can be used to train multiple models with different configurations if desired; however, the target, target type, and primary metric should not be changed after initialization so that models trained by this pipeline can be compared. 



**Example:**
 ```python
data = pd.read_csv(...)
pipeline = Pipeline(features, target, TargetType.CLASSIFICATION)
trained_model = pipeline.train(data)
``` 



**Attributes:**
 
 - <b>`name`</b>:  The name of the pipeline. 
 - <b>`target`</b>:  The name of the target column. 
 - <b>`target_type`</b>:  The type of the target column. 
 - <b>`primary_metric`</b>:  The primary metric to use for training and evaluation. 
 - <b>`feature_configs`</b>:  A dictionary mapping feature names to feature configurations. 
 - <b>`shuffle_data`</b>:  Whether or not to shuffle the data before training. 
 - <b>`drop_empty_percentage`</b>:  The percentage of empty values in a single row that will  result in dropping the row before training. 
 - <b>`dataset_split`</b>:  The dataset split to use for training and evaluation. 
 - <b>`configs`</b>:  A dictionary mapping versioned pipeline configuration IDs to pipeline  configurations. 
 - <b>`datasets`</b>:  A dictionary mapping versioned dataset IDs to datasets. 
 - <b>`uuid`</b>:  The UUID of the pipeline. This will be `None` unless the pipeline has been  published. 
 - <b>`trained_models`</b>:  A dictionary mapping trained model UUIDs to trained models. 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    features: 'List[str]',
    target: 'str',
    target_type: 'TargetType',
    categories: 'Optional[Dict[str, Union[List[int], List[str]]]]' = None,
    primary_metric: 'Optional[Metric]' = None,
    name: 'Optional[str]' = None,
    default_allow_hosting: 'bool' = True
)
```

Initializes an instance of `Pipeline`. 

The pipeline is initialized with a default config, which can be modified later. The target type can be optionally specfified. The default primary metric will be AUC for classification and Mean Squared Error for regression if not specified. 



**Args:**
 
 - <b>`features`</b>:  The column names in your data to use as features. 
 - <b>`target`</b>:  The name of the target column. 
 - <b>`target_type`</b>:  The type of the target column. 
 - <b>`categories`</b>:  A dictionary mapping feature names to unique categories. Any  values not in the categories list for a given feature will be treated  as a missing value. 
 - <b>`primary_metric`</b>:  The primary metric to use for training and evaluation. 
 - <b>`name`</b>:  The name of the pipeline. If not provided, the name will be set to  `{target}_{target_type}`. 
 - <b>`default_allow_hosting`</b>:  Whether or not to allow hosting by default for  datasets and models created by this pipeline. 




---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L374"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `analysis`

```python
analysis(trained_model: 'TrainedModel') → Optional[str]
```

Charts the results for the specified trained model in the SOTAI web client. 

This function requires an internet connection and a SOTAI account. The trained model will be uploaded to the SOTAI web client for analysis. 

If you would like to analyze the results for a trained model without uploading it to the SOTAI web client, the data is available in `training_results`. 



**Args:**
 
 - <b>`trained_model`</b>:  The trained model to be analyzed. 



**Returns:**
 If the analysis was successfully run, the analysis URL. Otherwise `None`. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L460"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `await_inference`

```python
await_inference(inference_uuid: 'str', inference_results_folder_path: 'str')
```

Polls the SOTAI cloud for the results of the specified inference job. 



**Args:**
 
 - <b>`inference_uuid`</b>:  The uuid of the inference job to poll. 
 - <b>`inference_results_folder_path`</b>:  The path to save the inference results to. 



**Returns:**
 If inference was successfully run, the path to the inference results. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L631"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_config`

```python
from_config(config: 'PipelineConfig', name: 'Optional[str]' = None) → Pipeline
```

Returns a new pipeline created from the specified config. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L551"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `from_hosted`

```python
from_hosted(
    pipeline_uuid: 'str',
    include_trained_models: 'bool' = False,
    include_datasets: 'bool' = False
) → Pipeline
```

Returns a new pipeline created from the specified hosted pipeline uuid. 



**Args:**
 
 - <b>`pipeline_uuid`</b>:  The uuid of the hosted pipeline to load. 
 - <b>`include_trained_models`</b>:  If True, trained models will be loaded along with  the pipeline. If False (default), trained models will not be loaded. 



**Returns:**
 A `Pipeline` instance. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `hypertune`

```python
hypertune(
    data: 'Union[int, DataFrame]',
    hypertune_config: 'HypertuneConfig',
    pipeline_config_id: 'Optional[int]' = None,
    model_config: 'Optional[LinearConfig]' = None,
    hosted: 'bool' = False
) → List[Union[TrainedModel, str]]
```

Returns a list of trained models trained according to the given configs. 



**Args:**
 
 - <b>`data`</b>:  The raw data to be prepared and trained on. If an int is provided,  it is assumed to be a dataset id and the corresponding dataset will be  used. 
 - <b>`hypertune_config`</b>:  The config to be used for hypertuning the model. 
 - <b>`pipeline_config_id`</b>:  The id of the pipeline config to be used for training.  If not provided, the current pipeline config will be versioned and used.  If data is an int, this argument is ignored and the pipeline config used  to prepare the data with the given id will be used. 
 - <b>`model_config`</b>:  The config to be used for training the model. If not provided,  a default config will be used. 
 - <b>`hosted`</b>:  Whether to run the hypertune job on the SOTAI cloud. If False, the  hypertune job will be run locally. 



**Returns:**
 A list of `TrainedModel` instances if run locally, or a list of trained model uuids if run in the SOTAI cloud. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L428"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `inference`

```python
inference(filepath: 'str', trained_model: 'TrainedModel') → Optional[str]
```

Runs inference on a dataset with a trained model in the SOTAI cloud. 



**Args:**
 
 - <b>`inference_dataset_path`</b>:  The path to the dataset to run inference on. 
 - <b>`trained_model`</b>:  The trained model to use for inference. 



**Returns:**
 If UUID of the inference run. If unsuccessful, `None`. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L605"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(filepath: 'str') → Pipeline
```

Loads the pipeline from the specified filepath. 



**Args:**
 
 - <b>`filepath`</b>:  The filepath from which to load the pipeline. The filepath should  point to a file created by the `save` method of a `TrainedModel`  instance. 



**Returns:**
 A `Pipeline` instance. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prepare`

```python
prepare(
    data: 'DataFrame',
    pipeline_config_id: 'Optional[int]' = None
) → Tuple[Dataset, PipelineConfig]
```

Prepares the data and versions it along with the current pipeline config. 

If any features in data are detected as non-numeric, the pipeline will attempt to handle them as categorical features. Any features that the pipeline cannot handle will be skipped. 



**Args:**
 
 - <b>`data`</b>:  The raw data to be prepared for training. 
 - <b>`pipeline_config_id`</b>:  The id of the pipeline config to be used for training.  If not provided, the current pipeline config will be used and versioned. 



**Returns:**
 A tuple of the versioned dataset and pipeline config. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `publish`

```python
publish() → Optional[str]
```

Uploads the pipeline to the SOTAI web client. 



**Returns:**
  If the pipeline was successfully uploaded, the pipeline UUID.  Otherwise, None. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L496"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(filepath: 'str', include_trained_models: 'bool' = True)
```

Saves the pipeline to the specified filepath. 

Trained models will be saved with the pipeline by default. 



**Args:**
 
 - <b>`filepath`</b>:  The directory to which the pipeline wil be saved. If the directory  does not exist, this function will attempt to create it. If the  directory already exists, this function will overwrite any existing  content with conflicting filenames. 
 - <b>`include_trained_models`</b>:  If True (default), then all models trained under  this pipeline present in the trained_models dictionary attribute will  be stored along with the pipeline under a `trained_models/{id}`  directory. If False, trained models will be excluded from saving. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L523"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `sync`

```python
sync()
```

Syncs the pipeline with the SOTAI cloud. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/pipeline.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `train`

```python
train(
    data: 'Union[DataFrame, int]',
    pipeline_config_id: 'Optional[int]' = None,
    model_config: 'Optional[LinearConfig]' = None,
    training_config: 'Optional[TrainingConfig]' = None
) → TrainedModel
```

Returns a calibrated model trained according to the given configs. 



**Args:**
 
 - <b>`data`</b>:  The raw data to be prepared and trained on. If an int is provided,  it is assumed to be a dataset id and the corresponding dataset will be  used. 
 - <b>`pipeline_config_id`</b>:  The id of the pipeline config to be used for training.  If not provided, the current pipeline config will be versioned and used.  If data is an int, this argument is ignored and the pipeline config used  to prepare the data with the given id will be used. 
 - <b>`model_config`</b>:  The config to be used for training the model. If not provided,  a default config will be used. 
 - <b>`training_config`</b>:  The config to be used for training the model. If not  provided, a default config will be used. 



**Returns:**
 A `TrainedModel` instance. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
