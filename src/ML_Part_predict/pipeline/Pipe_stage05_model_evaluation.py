from src.ML_Part_predict.config.configuration import ConfigurationManager
from src.ML_Part_predict.components.model_evaluation import ModelEvaluationConfig
from src.ML_Part_predict import logger
from src.ML_Part_predict.components.model_evaluation import ModelEvaluation


STAGE_NAME="Model Evaluation Stage"

class ModelEvaluation_Training_Pipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(f"{STAGE_NAME} started...")
            config=ConfigurationManager()
            model_evaluationconfig=config.get_model_evaluation_config()
            model_evaluationconfig=ModelEvaluation(config=model_evaluationconfig)
            model_evaluationconfig.log_into_mlflow()
            logger.info(f" {STAGE_NAME} completed successfully... ")
        except Exception as e:
            raise e

if __name__=="__main__":
    try:
        logger.info(f"{STAGE_NAME} started....")
        dat=ModelEvaluation_Training_Pipeline()
        dat.main()
        logger.info(f"{STAGE_NAME} completed....")
    except Exception as e:
        raise e
    
