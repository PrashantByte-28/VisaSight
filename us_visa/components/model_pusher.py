import sys

from us_visa.exception import USvisaException
from us_visa.logger import logging
from us_visa.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.entity.local_estimator import USvisaEstimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of model evaluation stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.usvisa_estimator = USvisaEstimator(model_path=model_pusher_config.saved_model_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Copies the trained model into the project's saved_model folder
        so it can be committed and deployed (e.g. on Vercel).
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            self.usvisa_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=self.model_pusher_config.saved_model_path)

            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact
        except Exception as e:
            raise USvisaException(e, sys) from e