from src.ML_Part_predict.config.configuration import ConfigurationManager
from src.ML_Part_predict.components.data_transformation import (FeatureEngineering ,DataTransformation)
from src.ML_Part_predict import logger
from pathlib import Path

STAGE_NAME="Data Transformation Stage"

class DataTransformation_Training_Pipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"),"r") as f:
                status=f.read().split(" ")[-1]

            if status=="True":
                logger.info(f"{STAGE_NAME} started...")
                config=ConfigurationManager()
                getdatatransform=config.get_data_transformation_config()
                featureEng=FeatureEngineering(config=getdatatransform)
                featureEng.feature_E()
                D_transform=DataTransformation(config=getdatatransform)
                D_transform.train_test_split()
                logger.info(f" {STAGE_NAME} completed successfully... ")
            else:
                raise Exception("The data schema is not correct")
            
        except Exception as e:
            print(e)

if __name__=="__main__":
    try:
        logger.info(f"{STAGE_NAME} started....")
        dat=DataTransformation_Training_Pipeline()
        dat.main()
        logger.info(f"{STAGE_NAME} completed....")
    except Exception as e:
        raise e
    