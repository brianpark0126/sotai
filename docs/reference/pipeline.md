<a id="sotai/pipeline"></a>

# sotai/pipeline

A Pipeline for calibrated modeling.

<a id="sotai/pipeline.Pipeline"></a>

## Pipeline Objects

```python
class Pipeline(BaseModel)
```

A pipeline for calibrated modeling.

A pipline takes in raw data and outputs a calibrated model. This process breaks
down into the following steps:

- Cleaning. The raw data is cleaned according to the pipeline's cleaning config.
- Transforming. The cleaned data is transformed according to transformation configs.
- Preparation. The transformed data is split into train, val, and test sets.
- Training. Hypertune models on the train and val sets to find the best one.

You can then analyze trained models and their results, and you can use the best
model that you trust to make predictions on new data.

<a id="sotai/pipeline.Pipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(data: pd.DataFrame,
             target: str,
             target_type: Optional[TargetType] = None,
             primary_metric: Optional[Metric] = None,
             name: Optional[str] = None,
             categories: Optional[List[str]] = None)
```

Initializes an instance of `Pipeline`.

The pipeline is initialized with a default config, which can be modified later.
The target type can be optionally specfified. If not specified, the pipeline
will try to automatically determine the type of the target from the data. The
same is true for the primary metric. The default primary metric will be F1 score
for classification and Mean Squared Error for regression.

**Arguments**:

- `data` - The raw data to be used for training.
- `target` - The name of the target column.
- `target_type` - The type of the target column.
- `primary_metric` - The primary metric to use for training and evaluation.
- `name` - The name of the pipeline.
- `categories` - The column names in `data` for categorical columns.

<a id="sotai/pipeline.Pipeline.target"></a>

#### target

```python
def target()
```

Returns the target column.

<a id="sotai/pipeline.Pipeline.target_type"></a>

#### target\_type

```python
def target_type()
```

Returns the target type.

<a id="sotai/pipeline.Pipeline.primary_metric"></a>

#### primary\_metric

```python
def primary_metric()
```

Returns the primary metric.

<a id="sotai/pipeline.Pipeline.configs"></a>

#### configs

```python
def configs(config_id: int)
```

Returns the config with the given id.

<a id="sotai/pipeline.Pipeline.datasets"></a>

#### datasets

```python
def datasets(dataset_id: int)
```

Returns the data with the given id.

<a id="sotai/pipeline.Pipeline.models"></a>

#### models

```python
def models(model_id: int)
```

Returns the model with the given id.

<a id="sotai/pipeline.Pipeline.train"></a>

#### train

```python
def train(dataset_id: int, pipeline_config_id: int, model_config: ModelConfig,
          training_config: TrainingConfig) -> TrainedModel
```

Returns a model trained according to the model and training configs.

<a id="sotai/pipeline.Pipeline.hypertune"></a>

#### hypertune

```python
def hypertune(
        dataset_id: int, pipeline_config_id: int, model_config: ModelConfig,
        hypertune_config: HypertuneConfig) -> Tuple[int, float, List[int]]
```

Runs hyperparameter tuning for the pipeline according to the given config.

**Arguments**:

- `dataset_id` - The id of the dataset to be used for training.
- `pipeline_config_id` - The id of the pipeline config to be used for training.
- `model_config` - The config for the model to be trained.
- `hypertune_config` - The config for hyperparameter tuning.
  

**Returns**:

  A tuple of the best model id, the best model's primary metric, and a list of
  all model ids that were trained.

<a id="sotai/pipeline.Pipeline.run"></a>

#### run

```python
def run(
    dataset: Optional[Union[pd.DataFrame, int]] = None,
    pipeline_config_id: Optional[int] = None,
    prepare_data_config: Optional[PrepareDataConfig] = None,
    model_config: Optional[ModelConfig] = None,
    hypertune_config: Optional[HypertuneConfig] = None
) -> Tuple[int, float, List[int]]
```

Runs the pipeline according to the pipeline and training configs.

The full pipeline run process is as follows:
- Prepare the data.
- Hypertune to find the best model for the current config.

When `data` is not specified, the pipeline will use the most recently used data
unless this is the first run, in which case it will use the data that was passed
in during initialization. When `model_config` is not specified, the pipeline will
use the default model config. When `hypertune_config` is not specified, the
pipeline will use the default hypertune config.

A call to `run` will create new dataset and pipeline config versions unless
explicit ids for previous versions are provided.

**Arguments**:

- `dataset` - The data to be used for training. Can be a pandas DataFrame
  containing new data or the id of a previously used dataset. If not
  specified, the pipeline will use the most recently used dataset unless
  this is the first run, in which case it will use the data that was
  passed in during initialization.
- `pipeline_config_id` - The id of the pipeline config to be used for training.
  If not specified, the pipeline will use the current settings for the
  primary pipeline config.
- `prepare_data_config` - The config for preparing the data.
- `model_config` - The config for the model to be trained.
- `hypertune_config` - The config for hyperparameter tuning.
  

**Returns**:

  A tuple of the best model id, the best model's primary metric, and a list of
  all model ids that were trained.

<a id="sotai/pipeline.Pipeline.predict"></a>

#### predict

```python
def predict(data: pd.DataFrame,
            model_id: int = None) -> Tuple[pd.DataFrame, str]
```

Runs pipeline without training to generate predictions for given data.

**Arguments**:

- `data` - The data to be used for prediction. Must have all columns used for
  training the model to be used.
- `model_id` - The id of the model to be used for prediction.
  

**Returns**:

  A tuple containing a dataframe with predictions and the new of the new
  column, which will be the name of the target column with a tag appended to
  it (e.g. target_prediction).

<a id="sotai/pipeline.Pipeline.analyze"></a>

#### analyze

```python
def analyze(model_id: int)
```

Charts pipeline model results for a specific model.

The following charts will be generated:
- Calibrator charts for each feature.
- Feature importance bar chart with feature statistics.

**Arguments**:

- `model_id` - The id of the model to be analyzed.

<a id="sotai/pipeline.Pipeline.save"></a>

#### save

```python
def save(filanem: str)
```

Saves the pipeline to a file.

**Arguments**:

- `filename` - The name of the file to save the pipeline to.

<a id="sotai/pipeline.Pipeline.load"></a>

#### load

```python
@classmethod
def load(cls, filename: str) -> Pipeline
```

Loads the pipeline from a file.

**Arguments**:

- `filename` - The name of the file to load the pipeline from.
  

**Returns**:

  An instance of `Pipeline` loaded from the file.

