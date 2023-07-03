"""This module contains the API functions for interacting with the SOTAI API.""" ""
import os
from typing import Dict, Union
import logging
import requests
from typing import Optional
from .constants import SOTAI_API_ENDPOINT
from .types import (
    CategoricalFeatureConfig,
    FeatureType,
    NumericalFeatureConfig,
    PipelineConfig,
)


def set_api_key(api_key: str):
    """Set the SOTAI API key in the environment variables.

    Args:
        api_key: The API key to set.
    """
    os.environ["SOTAI_API_KEY"] = api_key


def get_api_key() -> str:
    """Get the SOTAI API key from the environment variables.

    Returns:
        The API key retrieved from the environment variable.
    """
    return os.environ["SOTAI_API_KEY"]


def get_auth_headers() -> Dict[str, str]:
    """Get the authentication headers for a pipeline.

    Returns:
        The authentication headers.
    """
    return {
        "sotai-api-key": get_api_key(),
    }


def post_pipeline(pipeline) -> Optional[str]:
    """Create a new pipeline on the SOTAI API .

    Args:
        pipeline: The pipeline to create.

    Returns:
        If created, the UUID of the created pipeline. Otherwise None.
    """
    response = requests.post(
        f"{SOTAI_API_ENDPOINT}/api/v1/pipelines",
        json={
            "name": pipeline.name,
            "target": pipeline.target,
            "target_column_type": pipeline.target_type,
            "primary_metric": pipeline.primary_metric,
        },
        headers=get_auth_headers(),
        timeout=10,
    )

    if response.status_code != 200:
        print("Failed to create pipeline")
        logging.info("Failed to create pipeline")
        return None

    return response.json()["uuid"]


def post_pipeline_config(
    pipeline_uuid: str, pipeline_config: PipelineConfig
) -> Optional[str]:
    """Create a new pipeline config on the SOTAI API .

    Args:
        pipeline_uuid: The pipeline uuid to create the pipeline config for.
        pipeline_config : The pipeline config to create.

    Returns:
        If created, the UUID of the created pipeline config. Otherwise None.
    """
    response = requests.post(
        f"{SOTAI_API_ENDPOINT}/api/v1/pipelines/{pipeline_uuid}/pipeline-configs",
        json={
            "shuffle_data": pipeline_config.shuffle_data,
            "drop_empty_percentage": pipeline_config.drop_empty_percentage,
            "train_percentage": pipeline_config.dataset_split.train,
            "validation_percentage": pipeline_config.dataset_split.val,
            "test_percentage": pipeline_config.dataset_split.test,
        },
        headers=get_auth_headers(),
        timeout=10,
    )

    if response.status_code != 200:
        print("Failed to create pipeline config")
        logging.info("Failed to create pipeline config")
        return None

    return response.json()["uuid"]


def post_pipeline_feature_configs(
    pipeline_config_uuid: str,
    feature_configs: Dict[str, Union[CategoricalFeatureConfig, NumericalFeatureConfig]],
) -> Optional[str]:
    """Create a new pipeline feature configs on the SOTAI API .

    Args:
        pipeline_config_uuid: The pipeline config uuid to create the pipeline feature configs for.
        feature_configs: The feature configs to create.

    Returns:
        If created, the UUID of the pipeline config. Otherwise None.

    """
    sotai_feature_configs = []

    for feature_config in feature_configs.values():
        sotai_feature_config = {
            "feature_name": feature_config.name,
            "feature_type": feature_config.type,
        }
        if feature_config.type == FeatureType.CATEGORICAL:
            if isinstance(feature_config.categories[0], int):
                sotai_feature_config["categories_int"] = feature_config.categories
            else:
                sotai_feature_config["categories_str"] = feature_config.categories
        else:
            sotai_feature_config["num_keypoints"] = feature_config.num_keypoints
            sotai_feature_config[
                "input_keypoints_init"
            ] = feature_config.input_keypoints_init
            sotai_feature_config[
                "input_keypoints_type"
            ] = feature_config.input_keypoints_type
            sotai_feature_config["monotonicity"] = feature_config.monotonicity

        sotai_feature_configs.append(sotai_feature_config)

    response = requests.post(
        f"{SOTAI_API_ENDPOINT}/api/v1/pipeline-configs/{pipeline_config_uuid}/feature-configs",
        json=sotai_feature_configs,
        headers=get_auth_headers(),
        timeout=10,
    )

    if response.status_code != 200:
        print("Failed to create pipeline feature configs")
        logging.info("Failed to create pipeline feature configs")
        return None

    return response.json()["uuid"]


def post_trained_model_analysis(
    pipeline_config_uuid: str, trained_model
) -> Dict[str, str]:
    """Create a new trained model analysis on the SOTAI API .

    Args:
        pipeline_config_uuid: The pipeline config uuid to create the trained model
            analysis for.
        trained_model: The trained model to create.

    Returns:
        A dict containing the UUIDs of the resources created as well as a link that
        can be used to view the trained model analysis.

        Keys:
            - `trainedModelMetadataUuid`: The UUID of the trained model.
            - `modelConfigUuid`: The UUID of the model configuration.
            - `pipelineConfigUuid`: The UUID of the pipeline configuration.
            - `analysisUrl`: The URL of the trained model analysis.

    """
    training_results = trained_model.training_results
    train_primary_metrics = training_results.train_primary_metric_by_epoch
    val_primary_metrics = training_results.val_primary_metric_by_epoch
    overall_model_results_dict = {
        "epochs": trained_model.training_config.epochs,
        "batch_size": trained_model.training_config.batch_size,
        "learning_rate": trained_model.training_config.learning_rate,
        "runtime_in_seconds": training_results.training_time,
        "train_loss_per_epoch": training_results.train_loss_by_epoch,
        "train_primary_metric_per_epoch": train_primary_metrics,
        "validation_loss_per_epoch": training_results.val_loss_by_epoch,
        "validation_primary_metric_per_epoch": val_primary_metrics,
        "test_loss": training_results.test_loss,
        "test_primary_metric": training_results.test_primary_metric,
        "feature_names": [
            feature.feature_name for feature in trained_model.model.features
        ],
        "linear_coefficients": [
            training_results.linear_coefficients[feature.feature_name]
            for feature in trained_model.model.features
        ],
    }
    feature_analyses_list = [
        {
            "feature_name": feature.feature_name,
            "feature_type": feature.feature_type.value,
            "statistic_min": feature.min,
            "statistic_max": feature.max,
            "statistic_mean": feature.mean,
            "statistic_median": feature.median,
            "statistic_std": feature.std,
            "keypoints_outputs": feature.keypoints_outputs,
            "keypoints_inputs_categorical": feature.keypoints_inputs_categorical,
            "keypoints_inputs_numerical": feature.keypoints_inputs_numerical,
        }
        for feature in trained_model.training_results.feature_analyses.values()
    ]
    trained_model_metadata_dict = {
        "epochs": trained_model.training_config.epochs,
        "batch_size": trained_model.training_config.batch_size,
        "learning_rate": trained_model.training_config.learning_rate,
        "train_primary_metric": [training_results.train_primary_metric_by_epoch[-1]],
        "validation_primary_metric": [training_results.val_primary_metric_by_epoch[-1]],
        "test_primary_metric": training_results.test_primary_metric,
    }

    model_config_dict = {
        "model_framework": "pytorch",
        "model_type": "linear",
        "loss_type": trained_model.training_config.loss_type.value,
        "primary_metric": trained_model.pipeline_config.primary_metric.value,
        "target_column_type": trained_model.pipeline_config.target_type.value,
        "target_column": trained_model.pipeline_config.target,
        "model_config_name": "Model 1",
    }

    response = requests.post(
        f"{SOTAI_API_ENDPOINT}/api/v1/pipeline-configs/{pipeline_config_uuid}/analysis",
        json={
            "trained_model_metadata": trained_model_metadata_dict,
            "overall_model_results": overall_model_results_dict,
            "model_config": model_config_dict,
            "feature_analyses": feature_analyses_list,
        },
        headers=get_auth_headers(),
        timeout=10,
    )

    if response.status_code != 200:
        print("Failed to create trained model analysis")
        logging.info("Failed to create trained model analysis")
        return None

    return response.json()
