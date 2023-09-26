"""Shap utils
The MIT License (MIT)

Copyright (c) 2018 Scott Lundberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import scipy
from shap import Explanation
from shap.plots._utils import convert_ordering
from shap.utils._exceptions import DimensionError



def calculate_feature_importance(
    shapley_values: np.ndarray, feature_names: List[str]
) -> List[Dict[str, Any]]:
    """Calculates the feature importance from the shapley values.

    Args:
        shapley_values: The shapley values.
        feature_names: The feature names.

    Returns:
        A list of dictionaries containing the feature name and the feature importance value.
    """
    mean_abs_shap_values = np.mean(np.abs(shapley_values), axis=0)
    sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
    sorted_features = [
        {
            "feature": feature_names[i],
            "value": float(mean_abs_shap_values[i]),
        }
        for i in sorted_indices
    ]
    max_display = 10
    if len(sorted_features) > max_display:
        total_sum = 0
        count = 0
        for i in range(max_display - 1, len(sorted_features)):
            total_sum += sorted_features[i]["value"]
            count += 1
        sorted_features[max_display - 1] = {
            "feature": f"Sum of {count} other features",
            "value": total_sum,
        }
    return sorted_features[:max_display]


def calculate_beeswarm(  # pylint: disable-msg=too-many-locals,too-many-branches,too-many-statements
    features: pd.DataFrame, shapley_values: np.ndarray, target: str
) -> List[Dict[str, Any]]:
    """Calculates the beeswarm plot data.

    Args:
        features: The features.
        shapley_values: The shapley values.
        target: The target column.

    Returns:
        A list of dictionaries containing the shapley values, position, color, and name.

        Each dict contains the following fields:

            shaps: The shapley values.
            pos: The position.
            cmap: The color map.
            name: The name.
            vmin: The minimum value.
            vmax: The maximum value.
            c: The color.
    """
    reserved_columns = [target, "logits", "probability"]
    reserved_columns_to_drop = [
        col for col in reserved_columns if col in features.columns
    ]
    if len(reserved_columns_to_drop) > 0:
        features.drop(reserved_columns_to_drop, axis=1, inplace=True)
    feature_names = features.columns.tolist()
    max_display = 10
    color = None
    order = Explanation.abs.mean(0)

    res = []
    sv_shape = shapley_values.shape
    if len(sv_shape) == 1:
        emsg = (
            "The beeswarm plot does not support plotting a single instance, please "
            "pass an explanation matrix with many instances!"
        )
        raise ValueError(emsg)
    elif len(sv_shape) > 2:
        emsg = (
            "The beeswarm plot does not support plotting explanations with "
            "instances that have more than one dimension!"
        )
        raise ValueError(emsg)

    # we make a copy here, because later there are places that might modify this array
    values = np.copy(shapley_values)
    if scipy.sparse.issparse(features):
        features = features.toarray()
    # feature_names = shap_exp.feature_names
    # if out_names is None: # TODO: waiting for slicer support
    #     out_names = shap_exp.output_names

    order = convert_ordering(order, values)

    # feature index to category flag
    idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
    features = features.values

    num_features = values.shape[1]
    if features is not None:
        shape_msg = (
            "The shape of the shap_values matrix does not match the shape "
            "of the provided data matrix."
        )
        if num_features - 1 == features.shape[1]:
            shape_msg += (
                " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? If so, just pass shap_values[:,:-1]."
            )
            raise DimensionError(shape_msg)
        if num_features != features.shape[1]:
            raise DimensionError(shape_msg)

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(feature_names))]
    feature_order = convert_ordering(order, Explanation(np.abs(values)))
    # here we build our feature names, accounting for the fact that some features
    # might be merged together
    feature_inds = feature_order[:max_display]
    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features < len(values[0]):
        num_cut = np.sum(
            [
                len(orig_inds[feature_order[i]])
                for i in range(num_features - 1, len(values[0]))
            ]
        )
        values[:, feature_order[num_features - 1]] = np.sum(
            [
                values[:, feature_order[i]]
                for i in range(num_features - 1, len(values[0]))
            ],
            0,
        )
        feature_names[
            feature_order[num_features - 1]
        ] = f"Sum of {num_cut} other features"
    row_height = 0.4

    max_rows = 10000
    random_sample = False
    # make the beeswarm dots
    if len(values[:, 0]) > max_rows:
        random_sample = True
        random_indices_for_selection = np.random.choice(
            len(values[:, 0]), max_rows, replace=False
        )
    for pos, i in enumerate(reversed(feature_inds)):
        if random_sample:
            shaps = values[random_indices_for_selection, i]
            fvalues = (
                None if features is None else features[random_indices_for_selection, i]
            )
        else:
            shaps = values[:, i]
            fvalues = None if features is None else features[:, i]
        # Generate 1000 random unique indices
        shap_len = len(shaps)
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if fvalues is not None:
            fvalues = fvalues[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]:  # check categorical feature
                colored_feature = False
            else:
                fvalues = np.array(
                    fvalues, dtype=np.float64
                )  # make sure this can be numeric
        except Exception: # pylint: disable=broad-except
            colored_feature = False
        nbins = 100
        quant = np.round(
            nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)
        )
        inds = np.argsort(quant + np.random.randn(shap_len) * 1e-6)
        layer = 0
        last_bin = -1
        y_values = np.zeros(shap_len)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            y_values[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        y_values *= 0.9 * (row_height / np.max(y_values + 1))

        if fvalues is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(fvalues, 5)
            vmax = np.nanpercentile(fvalues, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(fvalues, 1)
                vmax = np.nanpercentile(fvalues, 99)
                if vmin == vmax:
                    vmin = np.min(fvalues)
                    vmax = np.max(fvalues)
            if vmin > vmax:  # fixes rare numerical precision issues
                vmin = vmax
            assert fvalues.shape[0] == len(
                shaps
            ), "Feature and SHAP matrices must have the same number of rows!"

            # plot the nan fvalues in the interaction feature as grey
            nan_mask = np.isnan(fvalues)

            # plot the non-nan fvalues colored by the trimmed feature value
            cvals = fvalues[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            res.append(
                {
                    "shaps": shaps[np.invert(nan_mask)].tolist(),
                    "pos": (pos + y_values[np.invert(nan_mask)]).tolist(),
                    "cmap": color,
                    "vmin": vmin,
                    "vmax": vmax,
                    "c": cvals.tolist(),
                    "name": feature_names[i],
                }
            )
        else:
            res.append(
                {
                    "shaps": shaps.tolist(),
                    "pos": (pos + y_values).tolist(),
                    "cmap": color if colored_feature else "#777777",
                    "name": feature_names[i],
                    "vmin": 0,
                    "vmax": 0,
                }
            )
    return res


def calculate_scatter(features: pd.DataFrame, shap_values: np.ndarray): # pylint: disable-msg=too-many-locals
    """Calculate scatter plot data for all possible feature combinations.

    Args:
        features: The features.
        shap_values: The shapley values.

    Returns:
        A list of dictionaries containing the scatter plot data for each feature combination.

        Each dict contains the following fields:

            primary_feature_name: The name of the primary feature.
            colorization_feature_name: The name of the colorization feature.
            x_values: The x values.
            y_values: The y values.
            colors: The colors.
            xmin: The minimum x value.
            xmax: The maximum x value.
            ymin: The minimum y value.
            ymax: The maximum y value.
            histogram: The histogram data.
            histogram_bin_edges: The histogram bin edges.
    """
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, shap_values.shape[0])

    num_features = shap_values.shape[1]
    scatter_plot_data_list = []

    for primary_index in range(num_features):
        for colorization_index in range(num_features):
            primary_feature_name = features.columns[primary_index]
            colorization_feature_name = features.columns[colorization_index]

            # skip if primary feature is the same as colorization feature
            if primary_feature_name == colorization_feature_name:
                continue

            # Extract primary feature values for x-axis
            oinds = np.arange(shap_values.shape[0])
            np.random.shuffle(oinds)
            x_values = features.iloc[oinds, primary_index].values
            shap_value = shap_values[oinds, primary_index]
            order = np.argsort(-shap_value)

            # Sort the values and handle NaNs
            x_values = x_values[order]
            shap_value = shap_value[order]
            try:
                xv_nan = np.isnan(x_values)
                xv_no_nan = x_values[~xv_nan]

                # Calculate colors based on SHAP values of colorization feature
                cvals = features.iloc[:, colorization_index].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = np.nanmean(cvals)
            except (TypeError, ValueError):
                continue

            # Plot limits
            xmin = np.nanmin(x_values) - (np.nanmax(x_values) - np.nanmin(x_values)) / 20
            xmax = np.nanmax(x_values) + (np.nanmax(x_values) - np.nanmin(x_values)) / 20
            y1min = np.nanmin(cvals_imp)
            y1max = np.nanmax(cvals_imp)

            # Histogram data
            xvals = np.unique(xv_no_nan)
            if (
                len(xvals) / len(xv_no_nan) < 0.2
                and len(xvals) < 75
                and np.max(xvals) < 75
                and np.min(xvals) >= 0
            ):
                bin_edges = np.arange(np.min(xvals) - 0.5, np.max(xvals) + 1.5) - 0.5
            else:
                if len(xv_no_nan) >= 500:
                    bin_edges = 50
                elif len(xv_no_nan) >= 200:
                    bin_edges = 20
                elif len(xv_no_nan) >= 100:
                    bin_edges = 10
                else:
                    bin_edges = 5
            hist_data, bin_edges_data = np.histogram(x_values[~np.isnan(x_values)], bin_edges)

            scatter_plot_data_list.append(
                {
                    "primary_feature_name": primary_feature_name,
                    "colorization_feature_name": colorization_feature_name,
                    "x_values": xv_no_nan.tolist(),
                    "y_values": shap_value.tolist(),
                    "colors": cvals_imp.tolist(),
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": y1min,
                    "ymax": y1max,
                    "histogram": hist_data.tolist(),
                    "histogram_bin_edges": bin_edges_data.tolist(),
                }
            )

    return scatter_plot_data_list
