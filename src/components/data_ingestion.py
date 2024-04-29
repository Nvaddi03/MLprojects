import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass # this will help you to create class variable

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('/Users/narasimharaovaddi/PycharmProjects/MLProject/src/artifacts', "train.csv" )
    test_data_path: str = os.path.join('/Users/narasimharaovaddi/PycharmProjects/MLProject/src/artifacts', "test.csv")
    raw_data_path: str = os.path.join('/Users/narasimharaovaddi/PycharmProjects/MLProject/src/artifacts', "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("----------Data ingestion started-----------")
        try:
            logging.info("----------Read data fro source -----------")
            df = pd.read_csv("/Users/narasimharaovaddi/PycharmProjects/MLProject/notebook/data/stud.csv")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)

            logging.info("----------- Train test split initiated -----------")

            train_set, test_set = train_test_split(df,test_size=0.2)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("----------- Train test split completed -----------")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__ =="__main__":
    obj = DataIngestion()
    X_train,X_test = obj.initiate_data_ingestion()

    data_transfrmation = DataTransformation()
    train_arr, test_arr,_ = data_transfrmation.initiate_data_transformation(X_train,X_test)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
    print("completed")



