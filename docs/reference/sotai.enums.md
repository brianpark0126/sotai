<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.enums`
Enum Classes for SOTAI SDK. 



---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `TargetType`
The type of target to predict. 


- CLASSIFICATION: classification target i.e. binary 0/1. 
- REGRESSION: regression target i.e. continuous float. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LossType`
The type of loss function to use. 


- BINARY_CROSSENTROPY: binary cross entropy loss. 
- HINGE: hinge loss. 
- HUBER: huber loss. 
- MAE: mean absolute error loss. 
- MSE: mean squared error loss. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Metric`
The type of metric to use. 


- AUC: area under the ROC curve. 
- MAE: mean absolute error. 
- MSE: mean squared error. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InputKeypointsInit`
Type of initialization to use for NumericalCalibrator input keypoints. 


- QUANTILES: initialize the input keypoints such that each segment will see the same  number of examples. 
- UNIFORM: initialize the input keypoints uniformly spaced in the feature range. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InputKeypointsType`
The type of input keypoints to use. 


- FIXED: the input keypoints will be fixed during initialization. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeatureType`
Type of feature. 


- UNKNOWN: a feature with a type that our system does not currently support. 
- NUMERICAL: a numerical feature that should be calibrated using an instance of  `NumericalCalibrator`. 
- CATEGORICAL: a categorical feature that should be calibrated using an instance of  `CategoricalCalibrator`. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NumericalCalibratorInit`
Type of kernel initialization to use for NumericalCalibrator. 


- EQUAL_HEIGHTS: initialize the kernel such that all segments have the same height. 
- EQUAL_SLOPES: initialize the kernel such that all segments have the same slope. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CategoricalCalibratorInit`
Type of kernel initialization to use for CategoricalCalibrator. 


- UNIFORM: initialize the kernel with uniformly distributed values. The sample range  will be [`output_min`, `output_max`] if both are provided. 
- CONSTANT: initialize the kernel with a constant value for all categories. This  value will be `(output_min + output_max) / 2` if both are provided. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Monotonicity`
Type of monotonicity constraint. 


- NONE: no monotonicity constraint. 
- INCREASING: increasing monotonicity i.e. increasing input increases output. 
- DECREASING: decreasing monotonicity i.e. increasing input decreases output. 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L136"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `APIStatus`
Status of API. 


- SUCCESS: API call was successful 
- ERROR: API call was unsuccessful 





---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/enums.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `InferenceConfigStatus`
Enum for InferenceConfig status. 


- FAILED: inference job failed. 
- INITIALIZING: inference job is initializing. 
- PREPARING: inference job is preparing to run. 
- RUNNING: inference job is running. 
- SUCCESS: inference job completed successfully. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
