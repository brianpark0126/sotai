import os
from typing import Dict, List, Optional, Tuple, Union

import requests

from .constants import SOTAI_API_ENDPOINT
from .features import NumericalFeature
from .types import (CategoricalFeatureConfig, DatasetSplit, FeatureAnalysis,
                    FeatureType, LinearConfig, LossType,
                    NumericalFeatureConfig, PipelineConfig, TargetType,
                    TrainingConfig, TrainingResults)

# SOTAI_API_ENDPOINT = "http://localhost:8000"


def set_api_key(api_key: str):
    """Set the SOTAI API key in the environment variables."""
    os.environ["SOTAI_API_KEY"] = api_key


def get_api_key():
    """Get the SOTAI API key from the environment variables."""
    return os.environ["SOTAI_API_KEY"]


def get_auth_headers():
    """Get the authentication headers for a pipeline.

    Args:
        pipeline (Pipeline): The pipeline to get the headers for.

    Returns:
        dict: The authentication headers.
    """
    return {
        "sotai-api-key": get_api_key(),
    }


def post_pipeline(pipeline):
    """Create a new pipeline on the SOTAI API .

    Args:
        pipeline (Pipeline): The pipeline to create.

    Returns:
        Pipeline: The created pipeline.
    """
    response = requests.post(
        f"{SOTAI_API_ENDPOINT}/api/v1/public/pipeline",
        json={
            "name": pipeline.name,
            "target": pipeline.target,
            "target_column_type": pipeline.target_type,
            "primary_metric": pipeline.primary_metric,
        },
        headers=get_auth_headers(),
    )
    return response.json()["uuid"]


def post_pipeline_config(pipeline_uuid: str, pipeline_config: PipelineConfig):
    """Create a new pipeline config on the SOTAI API .

    Args:
        pipeline_uuid (str): The pipeline uuid to create the pipeline config for.
        pipeline_config (PipelineConfig): The pipeline config to create.

        Returns:
            PipelineConfig: The created pipeline config.
    """

    response = requests.post(
        f"{SOTAI_API_ENDPOINT}/api/v1/public/pipeline/{pipeline_uuid}/config",
        json={
            "shuffle_data": pipeline_config.shuffle_data,
            "drop_empty_percentage": pipeline_config.drop_empty_percentage,
            "train_percentage": pipeline_config.dataset_split.train,
            "validation_percentage": pipeline_config.dataset_split.val,
            "test_percentage": pipeline_config.dataset_split.test,
        },
        headers=get_auth_headers(),
    )
    return response.json()["uuid"]


def post_pipeline_feature_configs(pipeline_config_uuid: str, feature_configs: Dict[
            str, Union[CategoricalFeatureConfig, NumericalFeatureConfig]
        ]):
    """Create a new pipeline feature configs on the SOTAI API .

        Args:
            pipeline_config_uuid (str): The pipeline config uuid to create the pipeline feature configs for.
            feature_configs (Dict[
                str, Union[CategoricalFeatureConfig, NumericalFeatureConfig]
            ]): The feature configs to create.

        Returns:
            PipelineConfigUUID: The created pipeline config.
            
    """
    
    sotai_feature_configs = []

    for feature_config in feature_configs:
        feature_config = {
            "feature_name": feature_config.name,
            "feature_type": feature_config.type,
        }
        if feature_config["feature_type"] == FeatureType.CATEGORICAL:
            if isinstance(feature_config["categories"][0], int):
                feature_config["categories_int"] = feature_config.categories
            else: 
                feature_config["categories_str"] = feature_config.categories
        else:
            feature_config["num_keypoints"] = feature_config.num_keypoints
            feature_config["input_keypoints_init"] = feature_config.input_keypoints_init
            feature_config["input_keypoints_type"] = feature_config.input_keypoints_type
            feature_config["monotonicity"] = feature_config.monotonicity
        
        sotai_feature_configs.append(feature_config)

    response = requests.post(
        f"{SOTAI_API_ENDPOINT}/api/v1/public/pipeline_config/{pipeline_config_uuid}/feature_configs",
        json=sotai_feature_configs,
        headers=get_auth_headers(),
    )
    return response.json()["uuid"]


def post_trained_model_analysis(pipeline_config_uuid: str, trained_model):
    """Create a new trained model analysis on the SOTAI API .

    Args:
        pipeline_config_uuid (str): The pipeline config uuid to create the trained model analysis for.
        trained_model (TrainedModel): The trained model to create.

    Returns:
        A dict containing the UUIDs of the resources created as well as a link that
        can be used to view the trained model analysis.
    """
    response = requests.post(
        f"{SOTAI_API_ENDPOINT}/api/v1/public/pipeline_config/{pipeline_config_uuid}/analysis",
        json={
            "trained_model_metadata": {
                "epochs": trained_model.training_config.epochs,
                "batch_size": trained_model.training_config.batch_size,
                "learning_rate": trained_model.training_config.learning_rate,
            },
            "overall_model_results": {
                "runtime_in_seconds": trained_model.training_results.training_time,
                "train_loss_per_epoch": trained_model.training_results.train_loss_by_epoch,
                "train_primary_metric_per_epoch": trained_model.training_results.train_primary_metric_by_epoch,
                "val_loss_per_epoch": trained_model.training_results.val_loss_by_epoch,
                "val_primary_metric_per_epoch": trained_model.training_results.val_primary_metric_by_epoch,
                "test_loss": trained_model.training_results.test_loss,
                "test_primary_metric": trained_model.training_results.test_primary_metric,
                "feature_names": [
                    feature.feature_name for feature in trained_model.model.features
                ],
                "linear_coefficients": trained_model.training_results.linear_coefficients,
            },
            "model_config": {
                "model_framework": "pytorch",
                "model_type": "linear",
                "loss_type": trained_model.training_config.loss_type,
                "primary_metric": trained_model.pipeline_config.primary_metric,
                "target_column_type": trained_model.pipeline_config.target_type,
                "target_column": trained_model.pipeline_config.target,
                "primary_metric": trained_model.pipeline_config.primary_metric,
            },
            "feature_analyses": [
                {
                    "feature_name": feature.feature_name,
                    "feature_type": feature.feature_type,
                    "statistic_min": feature.min,
                    "statistic_max": feature.max,
                    "statistic_mean": feature.mean,
                    "statistic_median": feature.median,
                    "statistic_std": feature.std,
                }
                for feature in trained_model.training_results.feature_analyses.values()
            ],
        },
        headers=get_auth_headers(),
    )

    return response.json()
