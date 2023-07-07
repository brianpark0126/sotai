"""A Trained Model created for a pipeline."""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field

from .data import CSVData, replace_missing_values
from .enums import TargetType
from .models import CalibratedLinear
from .types import LinearConfig, PipelineConfig, TrainingConfig, TrainingResults


class TrainedModel(BaseModel):
    """A trained calibrated model.

    This model is a container for a trained calibrated model that provides useful
    methods for using the model. The trained calibrated model is the result of running
    the `train` method of a `Pipeline` instance.

    Example:

    ```python
    data = pd.read_csv("data.csv")
    predictions = trained_model.predict(data)
    trained_model.analyze()
    ```
    """

    dataset_id: int = Field(...)
    pipeline_uuid: Optional[str] = None
    pipeline_config: PipelineConfig = Field(...)
    model_config: LinearConfig = Field(...)
    training_config: TrainingConfig = Field(...)
    training_results: TrainingResults = Field(...)
    model: CalibratedLinear = Field(...)
    metadata_uuid: Optional[str] = None
    analysis_url: Optional[str] = None

    class Config:  # pylint: disable=missing-class-docstring,too-few-public-methods
        arbitrary_types_allowed = True

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Returns predictions for the given data.

        Args:
            data: The data to be used for prediction. Must have all columns used for
                training the model to be used.

        Returns:
            If the target type is regression, a numpy array of predictions. If the
            target type is classification, a tuple containing a numpy array of
            predictions (logits) and a numpy array of probabilities.
        """
        data = data.loc[:, list(self.pipeline_config.feature_configs.keys())]
        data = replace_missing_values(data, self.pipeline_config.feature_configs)

        csv_data = CSVData(data)
        csv_data.prepare(self.model.features, None)
        inputs = list(csv_data.batch(csv_data.num_examples))[0]
        with torch.no_grad():
            predictions = self.model(inputs).numpy()

        if self.pipeline_config.target_type == TargetType.REGRESSION:
            return predictions

        return predictions, 1.0 / (1.0 + np.exp(-predictions))

    def save(self, filepath: str):
        """Saves the trained model to the specified directory.

        Args:
            filepath: The directory to save the trained model to. If the directory does
                not exist, this function will attempt to create it. If the directory
                already exists, this function will overwrite any existing content with
                conflicting filenames.
        """
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        with open(os.path.join(filepath, "trained_model_metadata.pkl"), "wb") as file:
            pickle.dump(self.dict(exclude={"model"}), file)
        model_path = f"{filepath}/trained_ptcm_model.pt"
        torch.save(self.model, model_path)

    @classmethod
    def load(cls, filepath: str) -> TrainedModel:
        """Loads a trained model from the specified filepath.

        Args:
            filepath: The filepath to load the trained model from. The filepath should
                point to a file created by the `save` method of a `TrainedModel`
                instance.

        Returns:
            A `TrainedModel` instance.
        """
        with open(os.path.join(filepath, "trained_model_metadata.pkl"), "rb") as file:
            trained_model_metadata = pickle.load(file)
        model_path = f"{filepath}/trained_ptcm_model.pt"
        model = torch.load(model_path)
        model.eval()

        return TrainedModel(**trained_model_metadata, model=model)
