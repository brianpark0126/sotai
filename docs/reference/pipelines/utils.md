<a id="sotai/pipelines/utils"></a>

# sotai/pipelines/utils

Utility functions for pipelines.

<a id="sotai/pipelines/utils.determine_target_type"></a>

#### determine\_target\_type

```python
def determine_target_type(data: np.ndarray) -> TargetType
```

Determines the type of target from its data.

<a id="sotai/pipelines/utils.determine_feature_types"></a>

#### determine\_feature\_types

```python
def determine_feature_types(
        data: np.ndarray,
        target: str,
        categories: Optional[List[str]] = None) -> Dict[str, FeatureType]
```

Determines the type of feature data.

<a id="sotai/pipelines/utils.generate_default_feature_configs"></a>

#### generate\_default\_feature\_configs

```python
def generate_default_feature_configs(
    data: pd.DataFrame, target: str, feature_types: Dict[str, FeatureType]
) -> Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]]
```

Generates default feature configs for the given data and target.

