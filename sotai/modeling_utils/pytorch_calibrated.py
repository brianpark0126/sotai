"""PyTorch Calibrated training utility functions."""
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
import pytorch_calibrated as ptcm
import shap
import torch
import torchmetrics
from pydantic import BaseModel

from ..constants import MISSING_CATEGORY_VALUE, MISSING_NUMERICAL_VALUE
from ..enums import FeatureType, LossType, Metric
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


class PTCMPerEpochResults(BaseModel):
    """Container for the per-epoch results of training a PyTorch Calibrated model."""

    train_loss_by_epoch: List[float]
    train_primary_metric_by_epoch: List[float]
    val_loss_by_epoch: List[float]
    val_primary_metric_by_epoch: List[float]


def _create_ptcm_feature_configs(
    features: Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]],
    train_csv_data: ptcm.data.CSVData,
) -> List[
    Union[ptcm.configs.CategoricalFeatureConfig, ptcm.configs.NumericalFeatureConfig]
]:
    """Returns a list of PyTorch Calibrated feature configs."""
    ptcm_feature_configs = []

    for feature_name, feature_config in features.items():
        if feature_config.type == FeatureType.CATEGORICAL:
            ptcm_feature_configs.append(
                ptcm.configs.CategoricalFeatureConfig(
                    feature_name=feature_name,
                    categories=feature_config.categories,
                    missing_input_value=MISSING_NUMERICAL_VALUE,
                )
            )
        else:  # FeatureType.NUMERICAL
            ptcm_feature_configs.append(
                ptcm.configs.NumericalFeatureConfig(
                    feature_name=feature_name,
                    data=train_csv_data(feature_name),
                    num_keypoints=feature_config.num_keypoints,
                    input_keypoints_init=feature_config.input_keypoints_init,
                    missing_input_value=MISSING_NUMERICAL_VALUE,
                    monotonicity=feature_config.monotonicity,
                )
            )

    return ptcm_feature_configs


def _create_ptcm_loss(loss_type: LossType) -> torch.nn.Module:
    """returns a Torch loss function from the given `LossType`."""
    if loss_type == LossType.BINARY_CROSSENTROPY:
        return torch.nn.BCEWithLogitsLoss()
    if loss_type == LossType.HINGE:
        return torch.nn.HingeEmbeddingLoss()
    if loss_type == LossType.HUBER:
        return torch.nn.HuberLoss()
    if loss_type == LossType.MAE:
        return torch.nn.L1Loss()
    if loss_type == LossType.MSE:
        return torch.nn.MSELoss()

    raise ValueError(f"Unknown loss type: {loss_type}")


def _create_ptcm_metric(metric: Metric) -> torchmetrics.Metric:
    """Returns a torchmetric Metric for the given `Metric`."""
    if metric == Metric.AUC:
        return torchmetrics.AUROC("binary")
    if metric == Metric.MAE:
        return torchmetrics.MeanAbsoluteError()
    if metric == Metric.MSE:
        return torchmetrics.MeanSquaredError()

    raise ValueError(f"Unknown metric: {metric}")


def _train_ptcm_model(  # pylint: disable=too-many-locals
    target: str,
    primary_metric: Metric,
    train_csv_data: ptcm.data.CSVData,
    val_csv_data: ptcm.data.CSVData,
    pipeline_config: PipelineConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> Tuple[
    ptcm.models.CalibratedLinear,
    PTCMPerEpochResults,
    torch.nn.Module,
    torchmetrics.Metric,
]:
    """Trains a PyTorch Calibrated model according to the given config."""
    ptcm_feature_configs = _create_ptcm_feature_configs(
        pipeline_config.features, train_csv_data
    )
    ptcm_model = ptcm.models.CalibratedLinear(
        ptcm_feature_configs,
        output_min=model_config.options.output_min,
        output_max=model_config.options.output_max,
        use_bias=model_config.options.use_bias,
        output_calibration_num_keypoints=model_config.options.output_calibration_num_keypoints
        if model_config.options.output_calibration
        else None,
    )

    optimizer = torch.optim.Adam(
        ptcm_model.parameters(recurse=True), training_config.learning_rate
    )
    loss_fn = _create_ptcm_loss(training_config.loss_type)
    metric_fn = _create_ptcm_metric(primary_metric)

    train_loss_by_epoch = []
    train_primary_metric_by_epoch = []
    val_loss_by_epoch = []
    val_primary_metric_by_epoch = []
    train_csv_data.prepare(ptcm_feature_configs, target)
    val_csv_data.prepare(ptcm_feature_configs, target)
    val_examples, val_targets = list(val_csv_data.batch(val_csv_data.num_examples))[0]
    for _ in range(training_config.epochs):
        train_prediction_tensors = []
        train_target_tensors = []
        for example_batch, target_batch in train_csv_data.batch(
            training_config.batch_size
        ):
            optimizer.zero_grad()
            outputs = ptcm_model(example_batch)
            train_prediction_tensors.append(outputs)
            train_target_tensors.append(target_batch)
            loss = loss_fn(outputs, target_batch)
            loss.backward()
            optimizer.step()
            ptcm_model.constrain()
        with torch.no_grad():
            predictions = torch.cat(train_prediction_tensors)
            targets = torch.cat(train_target_tensors)
            train_loss = loss_fn(predictions, targets)
            train_loss_by_epoch.append(train_loss.tolist())
            train_metric = metric_fn(predictions, targets)
            train_primary_metric_by_epoch.append(train_metric.tolist())
            val_predictions = ptcm_model(val_examples)
            val_loss = loss_fn(val_predictions, val_targets)
            val_loss_by_epoch.append(val_loss.tolist())
            val_metric = metric_fn(val_predictions, val_targets)
            val_primary_metric_by_epoch.append(val_metric.tolist())

    ptcm_per_epoch_results = PTCMPerEpochResults(
        train_loss_by_epoch=train_loss_by_epoch,
        train_primary_metric_by_epoch=train_primary_metric_by_epoch,
        val_loss_by_epoch=val_loss_by_epoch,
        val_primary_metric_by_epoch=val_primary_metric_by_epoch,
    )

    return ptcm_model, ptcm_per_epoch_results, loss_fn, metric_fn


def _extract_feature_analyses_from_ptcm_model(
    ptcm_model: ptcm.models.CalibratedLinear,
    features: Dict[str, Union[NumericalFeatureConfig, CategoricalFeatureConfig]],
    data: pd.DataFrame,
) -> Dict[str, FeatureAnalysis]:
    """Extracts feature statistics and calibration weights for each feature.

    Args:
        ptcm_model: A PyTorch Calibrated model.
        features: A mapping from feature name to feature config.
        data: The training + validation data for this model.

    Returns:
        A dictionary mapping feature name to `FeatureAnalysis` instance.
    """
    feature_analyses = {}

    for feature_name, calibrator in ptcm_model.calibrators.items():
        feature_config = features[feature_name]

        if feature_config.type == FeatureType.NUMERICAL:
            keypoints_inputs_numerical = [
                float(x) for x in calibrator.keypoints_inputs()
            ]
            keypoints_inputs_categorical = []
        else:
            keypoints_inputs_numerical = []
            keypoints_inputs = [feature_config.categories] + [MISSING_CATEGORY_VALUE]
            keypoints_inputs_categorical = [str(ki) for ki in keypoints_inputs]

        feature_analyses[feature_name] = FeatureAnalysis(
            feature_name=feature_name,
            feature_type=feature_config.type,
            min=float(np.min(data[feature_name].values)),
            max=float(np.max(data[feature_name].values)),
            mean=float(np.mean(data[feature_name].values)),
            median=float(np.median(data[feature_name].values)),
            std=float(np.std(data[feature_name].values)),
            keypoints_inputs_numerical=keypoints_inputs_numerical,
            keypoints_inputs_categorical=keypoints_inputs_categorical,
            keypoints_outputs=[float(x) for x in calibrator.keypoints_outputs()],
        )

    return feature_analyses


def _extract_feature_importances_from_ptcm_model(
    ptcm_model: ptcm.models.CalibratedLinear,
    x_val: List[np.ndarray],
) -> Dict[str, float]:
    """Extracts the feature importances for each feature using validation samples.

    The feature importances returned are simply the mean absolute value across each
    individual example since shapley values are calculated at the individual example
    level.

    Args:
        model: A PyTorch Calibrated model.
        x_val: The validation data used for validating the model results. This is
            what the explainer uses for producing sampled estimates.

    Returns:
        A dictionary mapping feature names to importances.
    """
    # Extract feature names from the calibration layers
    feature_names = [
        feature_config.feature_name for feature_config in ptcm_model.feature_configs
    ]

    # Restructure the data to be the correct shape and pull samples
    num_examples = np.shape(x_val)[0]
    sample_size = min(num_examples, 100)
    formatted_samples = np.take(
        x_val, np.random.choice(num_examples, sample_size), axis=0
    )
    explanation_size = min(num_examples, 500)
    formatted_explanations = np.take(
        x_val, np.random.choice(num_examples, explanation_size), axis=0
    )

    # Create our Explainer and determine our shapley values --> feature importances
    def predict(examples: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return ptcm_model(torch.from_numpy(examples).double()).numpy()

    explainer = shap.KernelExplainer(predict, formatted_samples)
    shap_values = explainer.shap_values(formatted_explanations, nsamples=500)[0]
    feature_importances = np.mean(np.absolute(shap_values), axis=0)

    return {
        feature_name: feature_importances[i]
        for i, feature_name in enumerate(feature_names)
    }


def train_and_evaluate_ptcm_model(  # pylint: disable=too-many-locals
    dataset_id: int,
    dataset: Dataset,
    target: str,
    primary_metric: Metric,
    pipeline_config_id: int,
    pipeline_config: PipelineConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> TrainedModel:
    """Trains a PyTorch Calibrated model according to the given config."""
    train_csv_data = ptcm.data.CSVData(dataset.prepared_data.train)
    val_csv_data = ptcm.data.CSVData(dataset.prepared_data.val)

    training_start_time = time.time()
    trained_ptcm_model, per_epoch_results, loss_fn, metric_fn = _train_ptcm_model(
        target,
        primary_metric,
        train_csv_data,
        val_csv_data,
        pipeline_config,
        model_config,
        training_config,
    )
    training_time = time.time() - training_start_time

    evaluation_start_time = time.time()
    test_csv_data = ptcm.data.CSVData(dataset.prepared_data.test)
    test_csv_data.prepare(trained_ptcm_model.feature_configs, target)
    x_test, y_test = list(test_csv_data.batch(test_csv_data.num_examples))[0]
    with torch.no_grad():
        evaluation_predictions = trained_ptcm_model(x_test)
        evaluation_loss = loss_fn(evaluation_predictions, y_test)
        evaluation_metric = metric_fn(evaluation_predictions, y_test)
    evaluation_time = time.time() - evaluation_start_time

    feature_analyses_extraction_start_time = time.time()
    feature_analyses = _extract_feature_analyses_from_ptcm_model(
        trained_ptcm_model,
        pipeline_config.features,
        pd.concat([train_csv_data.prepared_data, val_csv_data.prepared_data]),
    )
    feature_analyses_extraction_time = (
        time.time() - feature_analyses_extraction_start_time
    )

    feature_importance_extraction_start_time = time.time()
    feature_importances = _extract_feature_importances_from_ptcm_model(
        trained_ptcm_model, val_csv_data.prepared_data.values
    )
    feature_importance_extraction_time = (
        time.time() - feature_importance_extraction_start_time
    )

    training_results = TrainingResults(
        training_time=training_time,
        train_loss_by_epoch=per_epoch_results.train_loss_by_epoch,
        train_primary_metric_by_epoch=per_epoch_results.train_primary_metric_by_epoch,
        val_loss_by_epoch=per_epoch_results.val_loss_by_epoch,
        val_primary_metric_by_epoch=per_epoch_results.val_primary_metric_by_epoch,
        evaluation_time=evaluation_time,
        test_loss=evaluation_loss.tolist(),
        test_primary_metric=evaluation_metric.tolist(),
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
        model=trained_ptcm_model,
    )


def ptcm_model_predict(
    ptcm_model: ptcm.models.CalibratedLinear, data: pd.DataFrame
) -> np.ndarray:
    """Returns predictions for the given input data using the given model."""
    csv_data = ptcm.data.CSVData(data)
    csv_data.prepare(ptcm_model.feature_configs, None)
    inputs = list(csv_data.batch(csv_data.num_examples))[0]
    with torch.no_grad():
        return ptcm_model(inputs).numpy()
