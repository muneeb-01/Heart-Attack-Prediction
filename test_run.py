# test_run.py
from src.components.data_ingestion import DataIngestion
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print("✅ Data ingestion ran successfully.")
    print("Train file:", train_data)
    print("Test file:", test_data)
