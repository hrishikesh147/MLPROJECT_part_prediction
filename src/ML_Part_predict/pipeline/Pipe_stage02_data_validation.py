from src.ML_Part_predict.config.configuration import ConfigurationManager
from src.ML_Part_predict.components.data_validation import DataValidation
from src.ML_Part_predict import logger

STAGE_NAME="Data Validation Stage"

class DataValidation_Training_Pipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(f"{STAGE_NAME} started...")
            conf=ConfigurationManager()
            data_val=conf.get_data_validation_config()
            validation=DataValidation(data_val)
            validation.validate_all_columns()
            logger.info(f" {STAGE_NAME} completed successfully... ")
        except Exception as e:
            raise e

if __name__=="__main__":
    try:
        logger.info(f"{STAGE_NAME} started....")
        dat=DataValidation_Training_Pipeline()
        dat.main()
        logger.info(f"{STAGE_NAME} completed....")
    except Exception as e:
        raise e
    