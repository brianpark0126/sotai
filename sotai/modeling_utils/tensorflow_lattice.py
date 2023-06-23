"""TensorFlow Lattice training utility functions."""
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_lattice as tfl

from ..constants import MISSING_CATEGORY_VALUE, MISSING_NUMERICAL_VALUE
from ..enums import FeatureType, LossType, Metric, TargetType
from ..types import (
    CategoricalFeature,
    Dataset,
    FeatureAnalysis,
    ModelConfig,
    NumericalFeature,
    PipelineConfig,
    TrainingConfig,
    TrainingResults,
)


def prepare_tfl_data(
    data: pd.DataFrame,
    features: Dict[str, Union[CategoricalFeature, NumericalFeature]],
    target: Optional[str],
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    """Prepares a dataset for training a TensorFlow Lattice model."""
    x_data = data[list(features.keys())]
    y_data = data[target].values.astype("float32") if target else None
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

    return x_list, y_data, x_dict


def create_tfl_feature_configs(
    features: Dict[str, Union[CategoricalFeature, NumericalFeature]],
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


def create_tfl_loss(loss_type: LossType) -> tf.keras.losses.Loss:
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


def create_tfl_metric(metric: Metric) -> tf.keras.metrics.Metric:
    """Returns a Keras metric from the given `Metric`."""
    if metric == Metric.AUC:
        return tf.keras.metrics.AUC(from_logits=True, name=Metric.AUC)
    if metric == Metric.MAE:
        return tf.keras.metrics.MeanAbsoluteError(name=Metric.MAE)
    if metric == Metric.MSE:
        return tf.keras.metrics.MeanSquaredError(name=Metric.MSE)
    raise ValueError(f"Unknown metric: {metric}")


def create_tfl_model(
    tfl_feature_configs: List[tfl.configs.FeatureConfig],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    primary_metric: Metric,
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

    tfl_model = tfl.premade.CalibratedLinear(tfl_model_config)

    tfl_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=training_config.learning_rate),
        loss=create_tfl_loss(training_config.loss_type),
        metrics=[create_tfl_metric(primary_metric)],
    )

    return tfl_model


def train_tfl_model(
    target_type: TargetType,
    primary_metric: Metric,
    x_train: List[np.ndarray],
    y_train: np.ndarray,
    x_val: List[np.ndarray],
    y_val: np.ndarray,
    train_dict: Dict[str, np.ndarray],
    features: Dict[str, Union[CategoricalFeature, NumericalFeature]],
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> Tuple[tfl.premade.CalibratedLinear, tf.keras.callbacks.History]:
    """Train a TensorFlow Lattice model on the provided data under the given config."""
    tfl_feature_configs = create_tfl_feature_configs(features, train_dict)
    tfl_model = create_tfl_model(
        tfl_feature_configs,
        model_config,
        training_config,
        primary_metric,
        np.array(y_train).squeeze(),
        target_type == TargetType.CLASSIFICATION,
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


def extract_feature_analyses_from_tfl_model(
    trained_tfl_model: tfl.premade.CalibratedLinear,
    features: Dict[str, Union[CategoricalFeature, NumericalFeature]],
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
            Union[CategoricalFeature, NumericalFeature],
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


def extract_coefficients_from_tfl_linear_model(
    tfl_linear_model: tfl.premade.CalibratedLinear,
    features: List[str],
) -> Dict[str, float]:
    """Extracts coefficients from a TensorFlow Lattice `CalibratedLinear` model."""
    linear_coefficients = {}
    for layer in tfl_linear_model.layers:
        if layer.name == "tfl_linear_0":
            for feature_name, coefficient in zip(
                features, layer.kernel.numpy().flatten()
            ):
                linear_coefficients[feature_name] = coefficient
            if layer.use_bias:
                linear_coefficients["bias"] = layer.bias.numpy()

    return linear_coefficients


def train_and_evaluate_tfl_model(
    dataset: Dataset,
    target: str,
    target_type: TargetType,
    primary_metric: Metric,
    pipeline_config: PipelineConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> Tuple[tfl.premade.CalibratedLinear, TrainingResults,]:
    """Trains and evaluates TensorFlow Lattice model according to the given config."""
    x_train, y_train, train_dict = prepare_tfl_data(
        dataset.prepared_data.train, pipeline_config.features, target
    )
    x_val, y_val, _ = prepare_tfl_data(
        dataset.prepared_data.val, pipeline_config.features, target
    )

    training_start_time = time.time()
    trained_tfl_model, history = train_tfl_model(
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

    x_test, y_test, _ = prepare_tfl_data(
        dataset.prepared_data.test, pipeline_config.features, target
    )

    evaluation_start_time = time.time()
    evaluation_results = trained_tfl_model.evaluate(x_test, y_test, verbose=0)
    evaluation_time = time.time() - evaluation_start_time

    feature_analyses = extract_feature_analyses_from_tfl_model(
        trained_tfl_model,
        pipeline_config.features,
        np.concatenate([np.transpose(x_train), np.transpose(x_val)], axis=0).T,
    )

    linear_coefficients = extract_coefficients_from_tfl_linear_model(
        trained_tfl_model, list(pipeline_config.features.keys())
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
        feature_analyses=feature_analyses,
        linear_coefficients=linear_coefficients,
    )

    return trained_tfl_model, training_results
