from src.ML_Part_predict import logger
from src.ML_Part_predict.pipeline.Pipe_stage01_data_ingestion import DataIngestion_Training_pipeline

logger.info("welcome to custom logger")


STAGE_NAME= "Data Ingestion Stage"
try:
    logger.info(f"{STAGE_NAME} Started...")
    dat=DataIngestion_Training_pipeline()
    dat.main()
    logger.info(f"{STAGE_NAME} Completed...")
except Exception as e:
    raise e