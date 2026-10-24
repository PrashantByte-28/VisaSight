import sys

import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from us_visa.exception import USvisaException
from us_visa.logger import logging


# Approximate number of pay periods in a year, used to put every wage on the same scale.
WAGE_UNIT_TO_YEARLY = {"Hour": 2080, "Week": 52, "Month": 12, "Year": 1}


def convert_wage_to_yearly(dataframe: DataFrame) -> DataFrame:
    """
    Converts prevailing_wage to a yearly amount using unit_of_wage
    (Hour x 2080, Week x 52, Month x 12, Year x 1).
    Used for both training and prediction so the model always sees the same scale.
    """
    df = dataframe.copy()
    wage = pd.to_numeric(df["prevailing_wage"])
    factor = df["unit_of_wage"].map(WAGE_UNIT_TO_YEARLY).fillna(1)
    df["prevailing_wage"] = wage * factor
    return df


class TargetValueMapping:
    def __init__(self):
        self.Certified: int = 0
        self.Denied: int = 1

    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))



class USvisaModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model 
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        logging.info("Entered predict method of UTruckModel class")

        try:
            logging.info("Using the trained model to get predictions")

            dataframe = convert_wage_to_yearly(dataframe)
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get predictions")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise USvisaException(e, sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"