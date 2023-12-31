from src.ML_Part_predict import logger
from src.ML_Part_predict.pipeline.Pipe_stage01_data_ingestion import DataIngestion_Training_pipeline
from src.ML_Part_predict.pipeline.Pipe_stage02_data_validation import DataValidation_Training_Pipeline
from src.ML_Part_predict.pipeline.Pipe_stage03_data_transformation import DataTransformation_Training_Pipeline
from src.ML_Part_predict.pipeline.Pipe_stage04_model_training import ModelTrainer_Training_Pipeline
from src.ML_Part_predict.pipeline.Pipe_stage05_model_evaluation import ModelEvaluation_Training_Pipeline

STAGE_NAME= "Data Ingestion Stage"
try:
    logger.info(f"{STAGE_NAME} Started...")
    dat=DataIngestion_Training_pipeline()
    dat.main()
    logger.info(f"{STAGE_NAME} Completed...")
except Exception as e:
    raise e


STAGE_NAME="Data Validation Stage"
try:
    logger.info(f"{STAGE_NAME} started....")
    dat=DataValidation_Training_Pipeline()
    dat.main()
    logger.info(f"{STAGE_NAME} completed....")
except Exception as e:
    raise e


STAGE_NAME="Data Transformation Stage"
try:
    logger.info(f"{STAGE_NAME} started....")
    dat=DataTransformation_Training_Pipeline()
    dat.main()
    logger.info(f"{STAGE_NAME} completed....")
except Exception as e:
    raise e


STAGE_NAME="Model Training Stage"
try:
    logger.info(f"Stage : {STAGE_NAME} initiated...")
    obj=ModelTrainer_Training_Pipeline()
    obj.main()
    logger.info(f"Stage : {STAGE_NAME} completed...")
except Exception as e:
    raise e

STAGE_NAME="Model Evaluation Stage"
try:
    logger.info(f"Stage : {STAGE_NAME} initiated...")     
    dat=ModelEvaluation_Training_Pipeline()
    dat.main()    
    logger.info(f"Stage : {STAGE_NAME} completed...")
except Exception as e:
    raise e

