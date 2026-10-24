import os
import pickle
import shutil
import sys

from pandas import DataFrame

from us_visa.entity.estimator import USvisaModel
from us_visa.exception import USvisaException
from us_visa.logger import logging


class USvisaEstimator:
    """
    Saves, loads and serves the trained model from the local file system
    (the model file is bundled with the project, no cloud storage involved).
    """

    # Loaded models are cached per process so each request doesn't re-read the file.
    _model_cache = {}

    def __init__(self, model_path: str):
        """
        :param model_path: Local path of the saved model file
        """
        self.model_path = model_path
        self.loaded_model: USvisaModel = None

    def is_model_present(self) -> bool:
        return os.path.isfile(self.model_path)

    def load_model(self) -> USvisaModel:
        try:
            if self.model_path not in USvisaEstimator._model_cache:
                logging.info(f"Loading model from {self.model_path}")
                with open(self.model_path, "rb") as f:
                    USvisaEstimator._model_cache[self.model_path] = pickle.load(f)
            return USvisaEstimator._model_cache[self.model_path]
        except Exception as e:
            raise USvisaException(e, sys) from e

    def save_model(self, from_file: str) -> None:
        """
        Copy a trained model file to the project's saved model location.
        :param from_file: Local path of the newly trained model
        """
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            shutil.copyfile(from_file, self.model_path)
            USvisaEstimator._model_cache.pop(self.model_path, None)
        except Exception as e:
            raise USvisaException(e, sys) from e

    def predict(self, dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise USvisaException(e, sys) from e