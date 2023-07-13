<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/models.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.models`
PyTorch Calibrated Models to easily implement common calibrated model architectures. 

PyTorch Calibrated Models make it easy to construct common calibrated model architectures. To construct a PyTorch Calibrated Model, pass a calibrated modeling config to the corresponding calibrated model. 



---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/models.py#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CalibratedLinear`
PyTorch Calibrated Linear Model. 

Creates a `torch.nn.Module` representing a calibrated linear model, which will be constructed using the provided model configuration. Note that the model inputs should match the order in which they are defined in the `feature_configs`. 



**Attributes:**
 
    - All `__init__` arguments. 
 - <b>`calibrators`</b>:  A dictionary that maps feature names to their calibrators. 
 - <b>`linear`</b>:  The `Linear` layer of the model. 
 - <b>`output_calibrator`</b>:  The output `NumericalCalibrator` calibration layer. This  will be `None` if no output calibration is desired. 



**Example:**
 

```python
csv_data = CSVData(...)

feature_configs = [...]
calibrated_model = CalibratedLinear(feature_configs, ...)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(calibrated_model.parameters(recurse=True), lr=1e-1)

csv_data.prepare(feature_configs, "target", ...)
for epoch in range(100):
    for examples, targets in csv_data.batch(64):
         optimizer.zero_grad()
         outputs = calibrated_model(inputs)
         loss = loss_fn(outputs, labels)
         loss.backward()
         optimizer.step()
         calibrated_model.constrain()
``` 

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/models.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    features: List[Union[NumericalFeature, CategoricalFeature]],
    output_min: Optional[float] = None,
    output_max: Optional[float] = None,
    use_bias: bool = True,
    output_calibration_num_keypoints: Optional[int] = None
) → None
```

Initializes an instance of `CalibratedLinear`. 



**Args:**
 
 - <b>`features`</b>:  A list of numerical and/or categorical feature configs. 
 - <b>`output_min`</b>:  The minimum output value for the model. If `None`, the minimum  output value will be unbounded. 
 - <b>`output_max`</b>:  The maximum output value for the model. If `None`, the maximum  output value will be unbounded. 
 - <b>`use_bias`</b>:  Whether to use a bias term for the linear combination. If any of  `output_min`, `output_max`, or `output_calibration_num_keypoints` are  set, a bias term will not be used regardless of the setting here. 
 - <b>`output_calibration_num_keypoints`</b>:  The number of keypoints to use for the  output calibrator. If `None`, no output calibration will be used. 



**Raises:**
 
 - <b>`ValueError`</b>:  If any feature configs are not `NUMERICAL` or `CATEGORICAL`. 




---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/models/constrain#L177"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `constrain`

```python
constrain() → None
```

Constrains the model into desired constraints specified by the config. 

---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/models.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Tensor) → Tensor
```

Runs an input through the network to produce a calibrated linear output. 



**Args:**
 
 - <b>`x`</b>:  The input tensor of feature values of shape `(batch_size, num_features)`. 



**Returns:**
 torch.Tensor of shape `(batch_size, 1)` containing the model output result. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
