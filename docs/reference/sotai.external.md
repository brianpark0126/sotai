<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/external.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.external`
This module contains functions for external models to interact with the SOTAI API. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/external.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `shap`

```python
shap(
    inference_data: DataFrame,
    shapley_values: ndarray,
    base_values: ndarray,
    name: str,
    target: str,
    dataset_name: str
) → str
```

Uploads the shapley values, base values, and inference data to the SOTAI API. 



**Args:**
 
 - <b>`inference_data`</b>:  The data used for inference. 
 - <b>`shapley_values`</b>:  The shapley values for the inference data. 
 - <b>`base_values`</b>:  The base values for the inference data. 
 - <b>`name`</b>:  The name of the shapley values. 
 - <b>`target`</b>:  The target column of the inference data. 
 - <b>`dataset_name`</b>:  The name of the dataset used for inference. 



**Returns:**
 The UUID of the uploaded shapley values. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
