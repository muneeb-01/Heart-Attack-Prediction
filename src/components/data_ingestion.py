import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger_config import logger
from src.exception import Custom_Exception


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logger.info('Entered in my data ingestion method or component.')
        
        try:
            df = pd.read_csv("notebook/Data/heart.csv")
            logger.info("Read the data set as Data Frame.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logger.info("Train Test split Initiated")
            logger.info("Raw Data has been saved successfully in Artifacts folder.")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header = True)
            
            logger.info("Ingestion of the data is completed.")
            logger.info("Training and Test Data has been saved successfully in Artifacts folder.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logger.info("Error occurred during Data ingestion process.")
            raise Custom_Exception(e,sys)
        

if __name__ == "__main__":
    # For the function testing
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
