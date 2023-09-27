<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/utils/shap_utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.utils.shap_utils`
SHAP utility functions. 

Note that this code is based on code from the SHAP package, so we are including the license below: 

The MIT License (MIT) 

Copyright (c) 2018 Scott Lundberg 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: 

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/utils/shap_utils.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calculate_feature_importance`

```python
calculate_feature_importance(
    shapley_values: ndarray,
    feature_names: List[str]
) → List[Dict[str, Any]]
```

Calculates the feature importance from the shapley values. 



**Args:**
 
 - <b>`shapley_values`</b>:  The shapley values. 
 - <b>`feature_names`</b>:  The feature names. 



**Returns:**
 A list of dictionaries containing the feature name and the feature importance value. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/utils/shap_utils.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calculate_beeswarm`

```python
calculate_beeswarm(
    features: DataFrame,
    shapley_values: ndarray,
    target: str
) → List[Dict[str, Any]]
```

Calculates the beeswarm plot data. 



**Args:**
 
 - <b>`features`</b>:  The features. 
 - <b>`shapley_values`</b>:  The shapley values. 
 - <b>`target`</b>:  The target column. 



**Returns:**
 A list of dictionaries containing the shapley values, position, color, and name. 

Each dict contains the following fields: 


 - <b>`shaps`</b>:  The shapley values. 
 - <b>`pos`</b>:  The position. 
 - <b>`cmap`</b>:  The color map. 
 - <b>`name`</b>:  The name. 
 - <b>`vmin`</b>:  The minimum value. 
 - <b>`vmax`</b>:  The maximum value. 
 - <b>`c`</b>:  The color. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/utils/shap_utils.py#L282"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calculate_scatter`

```python
calculate_scatter(features: DataFrame, shap_values: ndarray)
```

Calculate scatter plot data for all possible feature combinations. 



**Args:**
 
 - <b>`features`</b>:  The features. 
 - <b>`shap_values`</b>:  The shapley values. 



**Returns:**
 A list of dictionaries containing the scatter plot data for each feature combination. 

Each dict contains the following fields: 


 - <b>`primary_feature_name`</b>:  The name of the primary feature. 
 - <b>`colorization_feature_name`</b>:  The name of the colorization feature. 
 - <b>`x_values`</b>:  The x values. 
 - <b>`y_values`</b>:  The y values. 
 - <b>`colors`</b>:  The colors. 
 - <b>`xmin`</b>:  The minimum x value. 
 - <b>`xmax`</b>:  The maximum x value. 
 - <b>`ymin`</b>:  The minimum y value. 
 - <b>`ymax`</b>:  The maximum y value. 
 - <b>`histogram`</b>:  The histogram data. 
 - <b>`histogram_bin_edges`</b>:  The histogram bin edges. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
