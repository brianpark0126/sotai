<a id="sotai/utils"></a>

# sotai/utils

Utility functions for pipelines.

<a id="sotai/utils.default_feature_config"></a>

#### default\_feature\_config

```python
def default_feature_config(feature_name: str, categorical: bool = False)
```

Generates a default feature config for the given feature.

**Arguments**:

- `feature_name` - The name of the feature.
- `categorical` - Whether the feature is categorical.
  

**Returns**:

  A default feature config for the given feature name.

<a id="sotai/utils.default_feature_configs"></a>

#### default\_feature\_configs

```python
def default_feature_configs(
    data: pd.DataFrame,
    target: str,
    categories: Optional[List[str]] = None
) -> Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]]
```

Generates default feature configs for the given data and target.

**Arguments**:

- `data` - The data to be used for training.
- `target` - The name of the target column.
- `feature_types` - A dictionary mapping column names to their corresponding feature
  type.
  

**Returns**:

  A dictionary mapping column names to their corresponding feature config.

