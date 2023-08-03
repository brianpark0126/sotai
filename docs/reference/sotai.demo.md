<!-- markdownlint-disable -->

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/demo.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `sotai.demo`
Functions for quickly prepping demo data to use with SOTAI. 


---

<a href="https://github.com/SOTAI-Labs/sotai/tree/main/sotai/demo.py#L7"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `heart`

```python
heart() → Tuple[DataFrame, List[str], str]
```

Prepares the demo heart dataset for use with the SOTAI Quickstart guide. 

The heart dataset is a classification dataset with 303 rows and 14 columns. The target is binary, with 0 indicating no heart disease and 1 indicating heart disease. The features are a mix of categorical and numerical features. For more information, see https://archive.ics.uci.edu/ml/datasets/heart+Disease. 



**Returns:**
  A tuple containing the cleaned heart dataset as a pandas `DataFrame`, the list  of features, and the target. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
