<a id="sotai/pipelines/types"></a>

# sotai/pipelines/types

Pydantic models for Pipelines.

<a id="sotai/pipelines/types.CleaningConfig"></a>

## CleaningConfig Objects

```python
class CleaningConfig(BaseModel)
```

Configuration for cleaning data.

**Attributes**:

- `drop_empty_percentage` - Drop rows that have drop_empty_percentage or more column
  values missing.

<a id="sotai/pipelines/types.NumericalFeatureConfig"></a>

## NumericalFeatureConfig Objects

```python
class NumericalFeatureConfig(_FeatureConfig)
```

Configuration for a numerical feature.

**Attributes**:

- `num_keypoints` - The number of keypoints to use for the calibrator.
- `input_keypoints_init` - The method for initializing the input keypoints.
- `input_keypoints_type` - The type of input keypoints.
- `monotonicity` - The monotonicity constraint, if any.

<a id="sotai/pipelines/types.CategoricalFeatureConfig"></a>

## CategoricalFeatureConfig Objects

```python
class CategoricalFeatureConfig(_FeatureConfig)
```

Configuration for a categorical feature.

**Attributes**:

- `categories` - The categories for the feature.
- `monotonicity_pairs` - The monotonicity constraints, if any, defined as pairs of
  categories. The output for the second category will be greater than or equal
  to the output for the first category for each pair, all else being equal.

<a id="sotai/pipelines/types.TransformationConfig"></a>

## TransformationConfig Objects

```python
class TransformationConfig(NumericalFeatureConfig)
```

Configuration for a transformation feature.

**Attributes**:

- `transformation_type` - The type of transformation.
- `primary_feature` - The name of the primary feature. This must match a column name
  in the dataset to be transformed.
- `secondary_feature` - The name of the secondary feature, if any, to use for the
  transformation. Either this or `secondary_value` must be provided for
  transformations that operate on two values.
- `secondary_value` - The secondary value, if any, to use for the transformation.
  Either this or `secondary_feature` must be provided for transformations that
  operate on two values.

<a id="sotai/pipelines/types.PipelineConfig"></a>

## PipelineConfig Objects

```python
class PipelineConfig(BaseModel)
```

A configuration object for a `Pipeline`.

**Attributes**:

- `cleaning_config` - The configuration to use for cleaning the dataset.
- `features` - A dictionary mapping the column name for a feature to its config.
- `transformations` - A dictionary mapping the column name for a feature
  transformation to its config.

<a id="sotai/pipelines/types.DatasetSplit"></a>

## DatasetSplit Objects

```python
class DatasetSplit(BaseModel)
```

Defines the split percentage for train, val, and test datasets.

**Attributes**:

- `train` - The percentage of the dataset to use for training.
- `val` - The percentage of the dataset to use for validation.
- `test` - The percentage of the dataset to use for testing.

<a id="sotai/pipelines/types.PreparedData"></a>

## PreparedData Objects

```python
class PreparedData(BaseModel)
```

A train, val, and test set of data that's been cleaned and transformed.

**Attributes**:

- `train` - The training dataset.
- `val` - The validation dataset.
- `test` - The testing dataset.

<a id="sotai/pipelines/types.Dataset"></a>

## Dataset Objects

```python
class Dataset(BaseModel)
```

A class for managing data.

**Attributes**:

- `raw_data` - The raw data.
- `dataset_split` - The split percentage for train, val, and test datasets.
- `prepared_data` - The prepared data.

<a id="sotai/pipelines/types.LinearOptions"></a>

## LinearOptions Objects

```python
class LinearOptions(_ModelOptions)
```

Calibrated Linear model options.

**Attributes**:

- `use_bias` - Whether to use a bias term for the linear combination.

<a id="sotai/pipelines/types.LatticeOptions"></a>

## LatticeOptions Objects

```python
class LatticeOptions(_ModelOptions)
```

Calibrated Lattice model options.

**Attributes**:

- `lattice_size` - The size of the lattice. For 1D lattices, this is the number of
  vertices. For higher-dimensional lattices, this is the number of vertices
  along each dimension. For example, a 2x2 lattice has 4 vertices.
- `interpolation` - The interpolation method to use for the lattice. Hypercube
  interpolation interpolates all vertices in the lattice. Simplex
  interpolation only interpolates the vertices along the edges of the lattice
  simplices.
- `parameterization` - The parameterization method to use for the lattice. Lattices
  with lattice size `L` and `N` inputs have ``L ** N`` parameters. All
  vertices parameterizes the lattice uses all ``L ** N`` vertices. KFL uses a
  factorized form that grows linearly with ``N``.
- `num_terms` - The number of terms to use for a kroncker-factored lattice. This will
  be ignored if the parameterization is not KFL.
- `random_seed` - The random seed to use for the lattice.

<a id="sotai/pipelines/types.EnsembleOptions"></a>

## EnsembleOptions Objects

```python
class EnsembleOptions(LatticeOptions)
```

Calibrated Lattice Ensemble model options.

**Attributes**:

- `lattices` - The type of ensembling to use for lattice arrangement.
- `num_lattices` - The number of lattices to use for the ensemble.
- `lattice_rank` - The number of features to use for each lattice in the ensemble.
- `separate_calibrators` - Whether to use separate calibrators for each lattice in
  the ensemble. If False, then a single calibrator will be used for each input
  feature.
- `use_linear_combination` - Whether to use a linear combination of the lattices in
  the ensemble. If False, then the output will be the average of the outputs.
- `use_bias` - Whether to use a bias term for the linear combination. Ignored if
  `use_linear_combination` is False.
- `fix_ensemble_for_2d_contraints` - Whether to fix the ensemble arrangement for 2D
  constraints.

<a id="sotai/pipelines/types.ModelConfig"></a>

## ModelConfig Objects

```python
class ModelConfig(BaseModel)
```

Configuration for a calibrated model.

**Attributes**:

- `framework` - The framework to use for the model (TensorFlow / PyTorch).
- `type` - The type of model to use.
- `options` - The configuration options for the model.

<a id="sotai/pipelines/types.TrainingConfig"></a>

## TrainingConfig Objects

```python
class TrainingConfig(BaseModel)
```

Configuration for training a single model.

**Attributes**:

- `loss_type` - The type of loss function to use for training.
- `epochs` - The number of iterations through the dataset during training.
- `batch_size` - The number of examples to use for each training step.
- `learning_rate` - The learning rate to use for the optimizer.

<a id="sotai/pipelines/types.HypertuneConfig"></a>

## HypertuneConfig Objects

```python
class HypertuneConfig(BaseModel)
```

Configuration for hyperparameter tuning to find the best model.

**Attributes**:

- `epochs_options` - A list of values to try for how many epochs to train the model.
- `batch_size_options` - A list of values to try for how many examples to use for
  each training step.
- `Linear_rate_options` - A list of values to try for the learning rate to use for
  the optimizer.

<a id="sotai/pipelines/types.FeatureAnalysis"></a>

## FeatureAnalysis Objects

```python
class FeatureAnalysis(BaseModel)
```

Feature analysis results for a single feature of a trained model.

**Attributes**:

- `feature_name` - The name of the feature.
- `feature_type` - The type of the feature.
- `min` - The minimum value of the feature.
- `max` - The maximum value of the feature.
- `mean` - The mean value of the feature.
- `median` - The median value of the feature.
- `std` - The standard deviation of the feature.
- `keypoints_inputs_numerical` - The input keypoints for the feature if the feature
  is numerical.
- `keypoints_inputs_categorical` - The input keypoints for the feature if the feature
  is categorical.
- `keypoints_outputs` - The output keypoints for each input keypoint.

<a id="sotai/pipelines/types.TrainingResults"></a>

## TrainingResults Objects

```python
class TrainingResults(BaseModel)
```

Training results for a single calibrated model.

**Attributes**:

- `training_time` - The total time spent training the model.
- `evaluation_time` - The total time spent evaluating the model.
- `feature_analysis_extraction_time` - The total time spent extracting feature
  analysis data from the model.
- `train_loss_by_epoch` - The training loss for each epoch.
- `train_primary_metric_by_epoch` - The training primary metric for each epoch.
- `validation_loss_by_epoch` - The validation loss for each epoch.
- `validation_primary_metric_by_epoch` - The validation primary metric for each
  epoch.
- `test_loss` - The test loss.
- `test_primary_metric` - The test primary metric.
- `feature_analysis_objects` - The feature analysis results for each feature.
- `feature_importances` - The feature importances for each feature.

<a id="sotai/pipelines/types.TrainedModel"></a>

## TrainedModel Objects

```python
class TrainedModel(BaseModel)
```

A calibrated model container for configs, results, and the model itself.

**Attributes**:

- `id` - The ID of the model.
- `model_config` - The configuration for the model.
- `training_config` - The configuration used for training the model.
- `training_results` - The results of training the model.
- `model` - The trained model.

