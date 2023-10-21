from src.ML_Part_predict.entity.config_entity import DataIngestionConfig
from src.ML_Part_predict.config.configuration import ConfigurationManager
from src.ML_Part_predict.components.data_ingestion import DataIngestion
from src.ML_Part_predict import logger

STAGE_NAME="data ingestion Stage"

class DataIngestion_Training_pipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config=ConfigurationManager()
            dataingestion_config=config.get_data_ingestion_config()
            dataingest=DataIngestion(dataingestion_config)
            dataingest.download_file()
            dataingest.extract_zip_file()
            
        except Exception as e:
            raise e


if __name__=="__main__":
    try:
        logger.info(f"{STAGE_NAME} Started...")
        dat=DataIngestion_Training_pipeline()
        dat.main()
        logger.info(f"{STAGE_NAME} Completed...")
    except Exception as e:
        raise e


