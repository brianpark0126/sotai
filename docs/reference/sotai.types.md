<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.types`
Pydantic models for Pipelines. 



---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DatasetSplit`
Defines the split percentage for train, val, and test datasets. 



**Attributes:**
 
 - <b>`train`</b>:  The percentage of the dataset to use for training. 
 - <b>`val`</b>:  The percentage of the dataset to use for validation. 
 - <b>`test`</b>:  The percentage of the dataset to use for testing. 




---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `validate_split_sum`

```python
validate_split_sum(values)
```

Ensures that the split percentages add up to 100. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PreparedData`
A train, val, and test set of data that's been cleaned. 



**Attributes:**
 
 - <b>`train`</b>:  The training dataset. 
 - <b>`val`</b>:  The validation dataset. 
 - <b>`test`</b>:  The testing dataset. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Dataset`
A class for managing data. 



**Attributes:**
 
 - <b>`pipeline_config_id`</b>:  The ID of the pipeline config used to create this dataset. 
 - <b>`prepared_data`</b>:  The prepared data ready for training. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L102"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LinearConfig`
Configuration for a calibrated linear model. 



**Attributes:**
 
 - <b>`use_bias`</b>:  Whether to use a bias term for the linear combination. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingConfig`
Configuration for training a single model. 



**Attributes:**
 
 - <b>`loss_type`</b>:  The type of loss function to use for training. 
 - <b>`epochs`</b>:  The number of iterations through the dataset during training. 
 - <b>`batch_size`</b>:  The number of examples to use for each training step. 
 - <b>`learning_rate`</b>:  The learning rate to use for the optimizer. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeatureAnalysis`
Feature analysis results for a single feature of a trained model. 



**Attributes:**
 
 - <b>`feature_name`</b>:  The name of the feature. 
 - <b>`feature_type`</b>:  The type of the feature. 
 - <b>`min`</b>:  The minimum value of the feature. 
 - <b>`max`</b>:  The maximum value of the feature. 
 - <b>`mean`</b>:  The mean value of the feature. 
 - <b>`median`</b>:  The median value of the feature. 
 - <b>`std`</b>:  The standard deviation of the feature. 
 - <b>`keypoints_inputs_numerical`</b>:  The input keypoints for the feature if the feature  is numerical. 
 - <b>`keypoints_inputs_categorical`</b>:  The input keypoints for the feature if the feature  is categorical. 
 - <b>`keypoints_outputs`</b>:  The output keypoints for each input keypoint. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TrainingResults`
Training results for a single calibrated model. 



**Attributes:**
 
 - <b>`training_time`</b>:  The total time spent training the model. 
 - <b>`train_loss_by_epoch`</b>:  The training loss for each epoch. 
 - <b>`train_primary_metric_by_epoch`</b>:  The training primary metric for each epoch. 
 - <b>`val_loss_by_epoch`</b>:  The validation loss for each epoch. 
 - <b>`val_primary_metric_by_epoch`</b>:  The validation primary metric for each  epoch. 
 - <b>`evaluation_time`</b>:  The total time spent evaluating the model. 
 - <b>`test_loss`</b>:  The test loss. 
 - <b>`test_primary_metric`</b>:  The test primary metric. 
 - <b>`feature_analyses`</b>:  The feature analysis results for each feature. 
 - <b>`linear_coefficients`</b>:  A mapping from feature name to linear coefficient. These  coefficients are the coefficients of the linear combination of features  after they have been calibrated, so any analysis of the coefficients should  be done with the feature's calibrator in mind. If using a bias term, the  bias value will be stored under the key "bias". Note that there will be no  bias term if you set output bounds or use an output calibrator. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NumericalFeatureConfig`
Configuration for a numerical feature. 



**Attributes:**
 
 - <b>`name`</b>:  The name of the feature. 
 - <b>`type`</b>:  The type of the feature. Always `FeatureType.NUMERICAL`. 
 - <b>`num_keypoints`</b>:  The number of keypoints to use for the calibrator. 
 - <b>`input_keypoints_init`</b>:  The method for initializing the input keypoints. 
 - <b>`input_keypoints_type`</b>:  The type of input keypoints. 
 - <b>`monotonicity`</b>:  The monotonicity constraint, if any. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CategoricalFeatureConfig`
Configuration for a categorical feature. 



**Attributes:**
 
 - <b>`name`</b>:  The name of the feature. 
 - <b>`type`</b>:  The type of the feature. Always `FeatureType.CATEGORICAL`. 
 - <b>`categories`</b>:  The categories for the feature. 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(**kwargs)
```









---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/types.py#L231"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PipelineConfig`
A configuration object for a `Pipeline`. 



**Attributes:**
 
 - <b>`id`</b>:  The ID of the pipeline config. This will be set by the Pipeline when it  versions this config during preparation. 
 - <b>`uuid`</b>:  The UUID of the pipeline. 
 - <b>`target`</b>:  The column name for the target. 
 - <b>`target_type`</b>:  The type of the target. 
 - <b>`primary_metric`</b>:  The primary metric to use for training and evaluation. 
 - <b>`feature_configs`</b>:  A dictionary mapping the column name for a feature to its  config. 
 - <b>`shuffle_data`</b>:  Whether to shuffle the data before splitting it into train,  validation, and test sets. 
 - <b>`drop_empty_percentage`</b>:  Rows will be dropped if they are this percentage empty. 
 - <b>`dataset_split`</b>:  The split of the dataset into train, validation, and test sets. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
