<a id="sotai/enums"></a>

# sotai/enums

Enum Classes for SOTAI SDK

<a id="sotai/enums.TargetType"></a>

## TargetType Objects

```python
class TargetType(_Enum)
```

The type of target to predict.

<a id="sotai/enums.LossType"></a>

## LossType Objects

```python
class LossType(_Enum)
```

The type of loss function to use.

<a id="sotai/enums.Metric"></a>

## Metric Objects

```python
class Metric(_Enum)
```

The type of metric to use.

<a id="sotai/enums.ModelFramework"></a>

## ModelFramework Objects

```python
class ModelFramework(_Enum)
```

The type of model framework to use.

<a id="sotai/enums.ModelType"></a>

## ModelType Objects

```python
class ModelType(_Enum)
```

The type of model to use.

<a id="sotai/enums.CalibratorRegularizationType"></a>

## CalibratorRegularizationType Objects

```python
class CalibratorRegularizationType(_Enum)
```

The type of regularization to use for the calibrator.

<a id="sotai/enums.LatticeRegularizationType"></a>

## LatticeRegularizationType Objects

```python
class LatticeRegularizationType(_Enum)
```

The type of regularization to use for the lattice.

<a id="sotai/enums.Interpolation"></a>

## Interpolation Objects

```python
class Interpolation(_Enum)
```

The type of interpolation to use for the lattice.

<a id="sotai/enums.Parameterization"></a>

## Parameterization Objects

```python
class Parameterization(_Enum)
```

The type of parameterization to use for the lattice.

<a id="sotai/enums.EnsembleType"></a>

## EnsembleType Objects

```python
class EnsembleType(_Enum)
```

The type of ensemble to use.

<a id="sotai/enums.InputKeypointsInit"></a>

## InputKeypointsInit Objects

```python
class InputKeypointsInit(_Enum)
```

Type of initialization to use for NumericalCalibrator input keypoints.

- QUANTILES: initialize the input keypoints such that each segment will see the same
    number of examples.
- UNIFORM: initialize the input keypoints uniformly spaced in the feature range.

<a id="sotai/enums.InputKeypointsType"></a>

## InputKeypointsType Objects

```python
class InputKeypointsType(_Enum)
```

The type of input keypoints to use.

<a id="sotai/enums.FeatureType"></a>

## FeatureType Objects

```python
class FeatureType(_Enum)
```

Type of feature.

- NUMERICAL: a numerical feature that should be calibrated using an instance of
    `NumericalCalibrator`.
- CATEGORICAL: a categorical feature that should be calibrated using an instance of
    `CategoricalCalibrator`.

<a id="sotai/enums.NumericalCalibratorInit"></a>

## NumericalCalibratorInit Objects

```python
class NumericalCalibratorInit(_Enum)
```

Type of kernel initialization to use for NumericalCalibrator.

- EQUAL_HEIGHTS: initialize the kernel such that all segments have the same height.
- EQUAL_SLOPES: initialize the kernel such that all segments have the same slope.

<a id="sotai/enums.CategoricalCalibratorInit"></a>

## CategoricalCalibratorInit Objects

```python
class CategoricalCalibratorInit(_Enum)
```

Type of kernel initialization to use for CategoricalCalibrator.

- UNIFORM: initialize the kernel with uniformly distributed values. The sample range
    will be [`output_min`, `output_max`] if both are provided.
- CONSTANT: initialize the kernel with a constant value for all categories. This
    value will be `(output_min + output_max) / 2` if both are provided.

<a id="sotai/enums.Monotonicity"></a>

## Monotonicity Objects

```python
class Monotonicity(_Enum)
```

Type of monotonicity constraint.

- NONE: no monotonicity constraint.
- INCREASING: increasing monotonicity i.e. increasing input increases output.
- DECREASING: decreasing monotonicity i.e. increasing input decreases output.

