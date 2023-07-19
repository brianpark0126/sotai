<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/trained_model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.trained_model`
A Trained Model created for a pipeline. 



---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/trained_model.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainedModel`
A trained calibrated model. 

This model is a container for a trained calibrated model that provides useful methods for using the model. The trained calibrated model is the result of running the `train` method of a `Pipeline` instance. 



**Example:**
 ```python
data = pd.read_csv("data.csv")
predictions = trained_model.predict(data)
trained_model.analyze()
``` 



**Attributes:**
 
 - <b>`dataset_id`</b>:  The ID of the dataset used to train the model. 
 - <b>`pipeline_uuid`</b>:  The UUID of the pipeline used to train the model. This will be  `None` if the trained model has not been analyzed under a pipeline. 
 - <b>`pipeline_config`</b>:  The configuration of the pipeline used to train the model. 
 - <b>`model_config`</b>:  The configuration of the model used to train the model. 
 - <b>`training_config`</b>:  The training configuration used to train the model. 
 - <b>`training_results`</b>:  The results of training the model. 
 - <b>`model`</b>:  The trained calibrated model. 
 - <b>`uuid`</b>:  The UUID of the trained model. This will be `None` if the trained model  has not been analyzed under a pipeline. 
 - <b>`analysis_url`</b>:  The URL of the analysis of the trained model. This will be `None`  if the trained model has not been analyzed under a pipeline. 




---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/trained_model.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(filepath: 'str') → TrainedModel
```

Loads a trained model from the specified filepath. 



**Args:**
 
 - <b>`filepath`</b>:  The filepath to load the trained model from. The filepath should  point to a file created by the `save` method of a `TrainedModel`  instance. 



**Returns:**
 A `TrainedModel` instance. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/trained_model.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(data: 'DataFrame') → Tuple[ndarray, Optional[ndarray]]
```

Returns predictions for the given data. 



**Args:**
 
 - <b>`data`</b>:  The data to be used for prediction. Must have all columns used for  training the model to be used. 



**Returns:**
 A tuple containing an array of predictions and an array of probabilities. If the target type is regression, then logits will be `None`. If the target type is classification, then the predictions will be logits. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/trained_model.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(filepath: 'str')
```

Saves the trained model to the specified directory. 



**Args:**
 
 - <b>`filepath`</b>:  The directory to save the trained model to. If the directory does  not exist, this function will attempt to create it. If the directory  already exists, this function will overwrite any existing content with  conflicting filenames. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
