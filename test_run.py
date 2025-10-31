# test_run.py
from src.components.data_ingestion import DataIngestion
from src.logger_config import logger
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # For the function testing
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    logger.info(f"Paths 1:{train_data}, 2:{test_data}")
    
    data_transformer = DataTransformation()
    train_arr, test_arr, _ = data_transformer.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    best_model_name, accuracy_score = model_trainer.initiate_model_trainer(train_arr,test_arr)
    
    logger.info("Everything is working fine together.")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Accuracy score: {accuracy_score}")
