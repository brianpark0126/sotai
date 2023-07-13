<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/training.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.training`
PyTorch Calibrated training utility functions. 

**Global Variables**
---------------
- **MISSING_CATEGORY_VALUE**
- **MISSING_NUMERICAL_VALUE**

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/training.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_features`

```python
create_features(
    feature_configs: Dict[str, Union[CategoricalFeatureConfig, NumericalFeatureConfig]],
    train_csv_data: CSVData
) → List[Union[NumericalFeature, CategoricalFeature]]
```

Returns a list of PyTorch Calibrated feature configs. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/training.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_loss`

```python
create_loss(loss_type: LossType) → Module
```

Returns a Torch loss function from the given `LossType`. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/training.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_metric`

```python
create_metric(metric: Metric) → Metric
```

Returns a torchmetric Metric for the given `Metric`. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/training.py#L96"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `create_model`

```python
create_model(
    features: List[Union[NumericalFeature, CategoricalFeature]],
    model_config: LinearConfig
)
```

Returns a PTCM model config constructed from the given `ModelConfig`. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/training.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_model`

```python
train_model(
    target: str,
    primary_metric: Metric,
    train_csv_data: CSVData,
    val_csv_data: CSVData,
    pipeline_config: PipelineConfig,
    model_config: LinearConfig,
    training_config: TrainingConfig
) → Tuple[CalibratedLinear, PerEpochResults, Module, Metric]
```

Trains a PyTorch Calibrated model according to the given config. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/training.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extract_feature_analyses`

```python
extract_feature_analyses(
    model: CalibratedLinear,
    feature_configs: Dict[str, Union[CategoricalFeatureConfig, NumericalFeatureConfig]],
    data: DataFrame
) → Dict[str, FeatureAnalysis]
```

Extracts feature statistics and calibration weights for each feature. 



**Args:**
 
 - <b>`ptcm_model`</b>:  A PyTorch Calibrated model. 
 - <b>`feature_configs`</b>:  A mapping from feature name to feature config. 
 - <b>`data`</b>:  The training + validation data for this model. 



**Returns:**
 A dictionary mapping feature name to `FeatureAnalysis` instance. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/training.py#L229"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `extract_linear_coefficients`

```python
extract_linear_coefficients(
    linear_model: CalibratedLinear,
    features: List[str]
) → Dict[str, float]
```

Extracts linear coefficients from a PyTorch `CalibratedLinear` model. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/training.py#L243"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `train_and_evaluate_model`

```python
train_and_evaluate_model(
    dataset: Dataset,
    target: str,
    primary_metric: Metric,
    pipeline_config: PipelineConfig,
    model_config: LinearConfig,
    training_config: TrainingConfig
) → Tuple[CalibratedLinear, TrainingResults]
```

Trains a PyTorch Calibrated model according to the given config. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/training.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PerEpochResults`
Container for the per-epoch results of training a PyTorch Calibrated model. 







---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
