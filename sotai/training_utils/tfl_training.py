"""TensorFlow Lattice training utility functions."""
import re
import time
import warnings
from typing import Dict, List, Tuple, Union

# pylint: disable=wrong-import-position
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", message=".*np.bool8` is a deprecated alias.*")
warnings.filterwarnings(
    "ignore", message=".*Importing display from IPython.core.display is deprecated.*"
)
# pylint: enable=wrong-import-position

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
import tensorflow_lattice as tfl

from ..constants import MISSING_CATEGORY_VALUE, MISSING_NUMERICAL_VALUE
from ..enums import FeatureType, LossType, Metric, TargetType
from ..types import (
    CategoricalFeatureConfig,
    Dataset,
    FeatureAnalysis,
    ModelConfig,
    NumericalFeatureConfig,
    PipelineConfig,
    TrainedModel,
    TrainingConfig,
    TrainingResults,
)


def _prepare_tfl_data(
    target: str,
    features: Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]],
    data: pd.DataFrame,
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    """Prepares a dataset for training a TensorFlow Lattice model."""
    x_data, y_data = (
        data[list(features.keys())],
        data[target],
    )
    x_list, x_dict = [], {}
    for feature_name, feature_config in features.items():
        values = x_data[feature_name].values
        if feature_config.type == FeatureType.CATEGORICAL:
            categories = feature_config.categories
            values = np.array(
                [
                    categories.index(c) if c in categories else len(categories)
                    for c in x_data[feature_name]
                ]
            )
        values = values.astype("float32")
        x_list.append(values)
        x_dict[feature_name] = values

    return x_list, y_data.values.astype("float32"), x_dict


def _create_tfl_feature_configs(
    features: Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]],
    train_dict: Dict[str, np.ndarray],
) -> List[tfl.configs.FeatureConfig]:
    """Returns TFL feature configs using the provided feature configs."""
    feature_configs = []
    for feature_name, feature_config in features.items():
        parameters = {"name": feature_name}

        if feature_config.type == FeatureType.CATEGORICAL:
            vocabulary = feature_config.categories
            num_buckets = len(vocabulary) + 1
            parameters["default_value"] = num_buckets
            parameters["num_buckets"] = num_buckets
            parameters["vocabulary_list"] = vocabulary
        else:  # FeatureType.NUMERICAL
            parameters["default_value"] = MISSING_NUMERICAL_VALUE
            parameters["monotonicity"] = feature_config.monotonicity
            parameters["pwl_calibration_num_keypoints"] = feature_config.num_keypoints
            parameters[
                "pwl_calibration_input_keypoints"
            ] = feature_config.input_keypoints_init
            parameters[
                "pwl_calibration_input_keypoints_type"
            ] = feature_config.input_keypoints_type

        feature_configs.append(tfl.configs.FeatureConfig(**parameters))

    # Finalize the feature configs
    feature_keypoints = tfl.premade_lib.compute_feature_keypoints(
        feature_configs=feature_configs, features=train_dict
    )
    tfl.premade_lib.set_feature_keypoints(
        feature_configs=feature_configs,
        feature_keypoints=feature_keypoints,
        add_missing_feature_configs=False,
    )

    return feature_configs


def _create_tfl_model(
    model_config: ModelConfig,
    tfl_feature_configs: List[tfl.configs.FeatureConfig],
    labels: np.ndarray,
    logits_output: bool,
) -> tfl.premade.CalibratedLinear:
    """Returns a TFL model config constructed from the given `ModelConfig`."""
    tfl_model_config = tfl.configs.CalibratedLinearConfig(
        feature_configs=tfl_feature_configs, **model_config.options.dict()
    )
    if model_config.options.output_calibration:
        label_keypoints = tfl.premade_lib.compute_label_keypoints(
            tfl_model_config, labels, logits_output
        )
        tfl.premade_lib.set_label_keypoints(tfl_model_config, label_keypoints)
    else:
        # This does nothing but is required to pass the model construction validation...
        tfl_model_config.output_initialization = [-2.0, 2.0]

    return tfl.premade.CalibratedLinear(tfl_model_config)


def _create_tfl_loss(loss_type: LossType) -> tf.keras.losses.Loss:
    """Returns a Keras loss function from the given `LossType`."""
    if loss_type == LossType.BINARY_CROSSENTROPY:
        return tf.keras.losses.BinaryCrossentropy()
    if loss_type == LossType.HINGE:
        return tf.keras.losses.Hinge()
    if loss_type == LossType.HUBER:
        return tf.keras.losses.Huber()
    if loss_type == LossType.MAE:
        return tf.keras.losses.MeanAbsoluteError()
    if loss_type == LossType.MSE:
        return tf.keras.losses.MeanSquaredError()
    raise ValueError(f"Unknown loss type: {loss_type}")


class FromLogitsMixin:  # pylint: disable=too-few-public-methods
    """TF Keras metric mixin to convert logits to probabilities."""

    def __init__(self, from_logits, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Standard update state method."""
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)


class F1Score(FromLogitsMixin, tf.keras.metrics.F1Score):
    """F1Score wrapper so that it works with logits."""


def _create_tfl_metric(metric: Metric) -> tf.keras.metrics.Metric:
    """Returns a Keras metric from the given `Metric`."""
    if metric == Metric.AUC:
        return tf.keras.metrics.AUC(from_logits=True)
    if metric == Metric.F1:
        return F1Score(threshold=0.5, from_logits=True)
    if metric == Metric.MAE:
        return tf.keras.metrics.MeanAbsoluteError()
    if metric == Metric.MSE:
        return tf.keras.metrics.MeanSquaredError()
    raise ValueError(f"Unknown metric: {metric}")


def _train_tfl_model(
    target_type: TargetType,
    primary_metric: Metric,
    x_train: List[np.ndarray],
    y_train: np.ndarray,
    x_val: List[np.ndarray],
    y_val: np.ndarray,
    train_dict: Dict[str, np.ndarray],
    features: Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]],
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> Tuple[tfl.premade.CalibratedLinear, tf.keras.callbacks.History]:
    """Train a TensorFlow Lattice model on the provided data under the given config."""
    tfl_feature_configs = _create_tfl_feature_configs(features, train_dict)
    tfl_model = _create_tfl_model(
        model_config,
        tfl_feature_configs,
        np.array(y_train).squeeze(),
        target_type == TargetType.CLASSIFICATION,
    )
    tfl_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_config.learning_rate),
        loss=_create_tfl_loss(training_config.loss_type),
        metrics=[_create_tfl_metric(primary_metric)],
    )
    history = tfl_model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=training_config.epochs,
        batch_size=training_config.batch_size,
        verbose=0,
    )
    return tfl_model, history


def _extract_feature_analyses_from_tfl_model(
    trained_tfl_model: Union[
        tfl.premade.CalibratedLinear,
        tfl.premade.CalibratedLattice,
        tfl.premade.CalibratedLatticeEnsemble,
    ],
    features: Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]],
    data: List[np.ndarray],
) -> Dict[str, FeatureAnalysis]:
    """Extracts feature statistics and calibration weights for each feature.

    Args:
        trained_tfl_model: A TensorFlow Lattice Premade model.
        features: A mapping from feature name to feature config.
        data: The training + validation data for this model.

    Returns:
        A dictionary mapping feature name to `FeatureAnalysis` instance.
    """
    feature_analyses = {}

    calibration_layers_info: Dict[
        str,
        Tuple[
            Union[NumericalFeatureConfig, CategoricalFeatureConfig],
            Union[tfl.layers.PWLCalibration, tfl.layers.CategoricalCalibration],
        ],
    ] = {}
    matcher = re.compile(r"(?<=tfl_calib_)\w+")
    for layer in trained_tfl_model.layers:
        match = matcher.match(layer.name)
        if not match:
            continue
        feature_name = match.group(0)
        calibration_layers_info[feature_name] = (features[feature_name], layer)

    for (feature_name, (feature_config, layer)), feature_data in zip(
        list(calibration_layers_info.items()), data
    ):
        if feature_config.type == FeatureType.NUMERICAL:
            keypoints_inputs_numerical = [
                float(x) for x in layer.keypoints_inputs().numpy().flatten()
            ]
            keypoints_inputs_categorical = []
            keypoints_outputs = [
                float(x) for x in layer.keypoints_outputs().numpy().flatten()
            ]
        else:
            keypoints_inputs_numerical = []
            keypoints_inputs_categorical = feature_config.categories
            keypoints_inputs_categorical.append(MISSING_CATEGORY_VALUE)
            keypoints_inputs_categorical = [
                str(ki) for ki in keypoints_inputs_categorical
            ]
            keypoints_outputs = [float(x) for x in layer.get_weights()[0].flatten()]

        feature_analyses[feature_name] = FeatureAnalysis(
            feature_name=feature_name,
            feature_type=feature_config.type,
            min=float(np.min(feature_data)),
            max=float(np.max(feature_data)),
            mean=float(np.mean(feature_data)),
            median=float(np.median(feature_data)),
            std=float(np.std(feature_data)),
            keypoints_inputs_numerical=keypoints_inputs_numerical,
            keypoints_inputs_categorical=keypoints_inputs_categorical,
            keypoints_outputs=keypoints_outputs,
        )

    return feature_analyses


def _extract_feature_importances_from_tfl_model(  # pylint: disable=too-many-locals
    model: Union[
        tfl.premade.CalibratedLinear,
        tfl.premade.CalibratedLattice,
        tfl.premade.CalibratedLatticeEnsemble,
    ],
    x_val: List[np.ndarray],
) -> Dict[str, float]:
    """Extracts the feature importances for each feature using validation samples.

    The feature importances returned are simply the mean absolute value across each
    individual example since shapley values are calculated at the individual example
    level.

    Args:
        model: A TensorFlow Lattice Premade model.
        x_val: The validation data used for validating the model results. This is
            what the explainer uses for producing sampled estimates.

    Returns:
        A dictionary mapping feature names to importances.
    """
    num_features = len(x_val)

    # Extract feature names from the calibration layers
    feature_names = []
    matcher = re.compile(r"(?<=tfl_calib_)\w+")
    for layer in model.layers:
        match = matcher.match(layer.name)
        if match:
            feature_names.append(match.group(0))

    # Wrap the model with the correct input shape
    wrapper_input = tf.keras.Input((num_features,))
    wrapper_output = model(tf.split(wrapper_input, num_features, axis=1))
    wrapper_model = tf.keras.Model(inputs=wrapper_input, outputs=wrapper_output)

    # Restructure the data to be the correct shape and pull samples
    formatted_data = np.transpose(x_val)
    num_examples = np.shape(formatted_data)[0]
    sample_size = min(num_examples, 100)
    formatted_samples = np.take(
        formatted_data, np.random.choice(num_examples, sample_size), axis=0
    )
    explanation_size = min(num_examples, 500)
    formatted_explanations = np.take(
        formatted_data, np.random.choice(num_examples, explanation_size), axis=0
    )

    # Create our Explainer and determine our shapley values --> feature importances
    explainer = shap.KernelExplainer(wrapper_model, formatted_samples)
    shap_values = explainer.shap_values(formatted_explanations, nsamples=500)[0]
    feature_importances = np.mean(np.absolute(shap_values), axis=0)

    return {
        feature_name: feature_importances[i]
        for i, feature_name in enumerate(feature_names)
    }


def train_and_evaluate_tfl_model(  # pylint: disable=too-many-locals
    dataset_id: int,
    dataset: Dataset,
    target: str,
    target_type: TargetType,
    primary_metric: Metric,
    pipeline_config_id: int,
    pipeline_config: PipelineConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> TrainedModel:
    """Trains and evaluates TensorFlow Lattice model according to the given config."""
    x_train, y_train, train_dict = _prepare_tfl_data(
        target, pipeline_config.features, dataset.prepared_data.train
    )
    x_val, y_val, _ = _prepare_tfl_data(
        target, pipeline_config.features, dataset.prepared_data.val
    )

    training_start_time = time.time()
    trained_tfl_model, history = _train_tfl_model(
        target_type,
        primary_metric,
        x_train,
        y_train,
        x_val,
        y_val,
        train_dict,
        pipeline_config.features,
        model_config,
        training_config,
    )
    training_time = time.time() - training_start_time

    x_test, y_test, _ = _prepare_tfl_data(
        target, pipeline_config.features, dataset.prepared_data.test
    )

    evaluation_start_time = time.time()
    evaluation_results = trained_tfl_model.evaluate(x_test, y_test, verbose=0)
    evaluation_time = time.time() - evaluation_start_time

    feature_analyses_extraction_start_time = time.time()
    feature_analyses = _extract_feature_analyses_from_tfl_model(
        trained_tfl_model,
        pipeline_config.features,
        np.concatenate([np.transpose(x_train), np.transpose(x_val)], axis=0).T,
    )
    feature_analyses_extraction_time = (
        time.time() - feature_analyses_extraction_start_time
    )

    feature_importance_extraction_start_time = time.time()
    feature_importances = _extract_feature_importances_from_tfl_model(
        trained_tfl_model, x_val
    )
    feature_importance_extraction_time = (
        time.time() - feature_importance_extraction_start_time
    )

    training_results = TrainingResults(
        training_time=training_time,
        train_loss_by_epoch=history.history["loss"],
        train_primary_metric_by_epoch=history.history[primary_metric],
        val_loss_by_epoch=history.history["val_loss"],
        val_primary_metric_by_epoch=history.history[f"val_{primary_metric}"],
        evaluation_time=evaluation_time,
        test_loss=evaluation_results[0],
        test_primary_metric=evaluation_results[1],
        feature_analyses_extraction_time=feature_analyses_extraction_time,
        feature_analyses=feature_analyses,
        feature_importance_extraction_time=feature_importance_extraction_time,
        feature_importances=feature_importances,
    )

    return TrainedModel(
        dataset_id=dataset_id,
        pipeline_config_id=pipeline_config_id,
        model_config=model_config,
        training_config=training_config,
        training_results=training_results,
        model=trained_tfl_model,
    )
