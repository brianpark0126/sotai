<a id="sotai/pipelines/types"></a>

# sotai/pipelines/types

Pydantic models for Pipelines.

<a id="sotai/pipelines/types.CleaningConfig"></a>

## CleaningConfig Objects

```python
class CleaningConfig(BaseModel)
```

Configuration for cleaning data.

<a id="sotai/pipelines/types.NumericalFeatureConfig"></a>

## NumericalFeatureConfig Objects

```python
class NumericalFeatureConfig(_FeatureConfig)
```

Configuration for a numerical feature.

<a id="sotai/pipelines/types.CategoricalFeatureConfig"></a>

## CategoricalFeatureConfig Objects

```python
class CategoricalFeatureConfig(_FeatureConfig)
```

Configuration for a categorical feature.

<a id="sotai/pipelines/types.TransformationConfig"></a>

## TransformationConfig Objects

```python
class TransformationConfig(NumericalFeatureConfig)
```

Configuration for a transformation feature.

<a id="sotai/pipelines/types.LinearOptions"></a>

## LinearOptions Objects

```python
class LinearOptions(_ModelOptions)
```

Calibrated Linear model options.

<a id="sotai/pipelines/types.LatticeOptions"></a>

## LatticeOptions Objects

```python
class LatticeOptions(_ModelOptions)
```

Calibrated Lattice model options.

<a id="sotai/pipelines/types.EnsembleOptions"></a>

## EnsembleOptions Objects

```python
class EnsembleOptions(LatticeOptions)
```

Calibrated Lattice Ensemble model options.

<a id="sotai/pipelines/types.ModelConfig"></a>

## ModelConfig Objects

```python
class ModelConfig(BaseModel)
```

Configuration for a calibrated model.

<a id="sotai/pipelines/types.TrainingConfig"></a>

## TrainingConfig Objects

```python
class TrainingConfig(BaseModel)
```

Configuration for training a single model.

<a id="sotai/pipelines/types.HypertuneConfig"></a>

## HypertuneConfig Objects

```python
class HypertuneConfig(BaseModel)
```

Configuration for hyperparameter tuning to find the best model.

<a id="sotai/pipelines/types.FeatureAnalysis"></a>

## FeatureAnalysis Objects

```python
class FeatureAnalysis(BaseModel)
```

Feature analysis results for a single feature of a trained model.

<a id="sotai/pipelines/types.TrainingResults"></a>

## TrainingResults Objects

```python
class TrainingResults(BaseModel)
```

Training results for a single calibrated model.

<a id="sotai/pipelines/types.Model"></a>

## Model Objects

```python
class Model(BaseModel)
```

A calibrated model container for configs, results, and the model itself.

<a id="sotai/pipelines/types.PipelineModels"></a>

## PipelineModels Objects

```python
class PipelineModels(BaseModel)
```

A container for the best model / metric and all models trained in a pipeline.

<a id="sotai/pipelines/types.PipelineConfig"></a>

## PipelineConfig Objects

```python
class PipelineConfig(BaseModel)
```

A configuration object for a `Pipeline`.

<a id="sotai/pipelines/types.DatasetSplit"></a>

## DatasetSplit Objects

```python
class DatasetSplit(BaseModel)
```

Defines the split percentage for train, val, and test datasets.

<a id="sotai/pipelines/types.PreparedData"></a>

## PreparedData Objects

```python
class PreparedData(BaseModel)
```

A train, val, and test set of data that's been cleaned and transformed.

<a id="sotai/pipelines/types.Data"></a>

## Data Objects

```python
class Data(BaseModel)
```

A class for managing data.

<a id="sotai/pipelines/types.PipelineData"></a>

## PipelineData Objects

```python
class PipelineData(BaseModel)
```

A class for managing pipeline data.

