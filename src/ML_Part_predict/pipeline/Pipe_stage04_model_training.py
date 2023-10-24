from src.ML_Part_predict.config.configuration import ConfigurationManager
from src.ML_Part_predict.components.model_trainer import ModelTrainer
from src.ML_Part_predict import logger

STAGE_NAME="Model Training"

class ModelTrainer_Training_Pipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            con=ConfigurationManager()
            model_trn=con.get_model_trainer_config()
            model_trn=ModelTrainer(model_trn)
            model_trn.train()
        except Exception as e:
            raise e
        
if __name__=="__main__":
    try:
        logger.info(f"Stage : {STAGE_NAME} initiated...")
        obj=ModelTrainer_Training_Pipeline()
        obj.main()
        logger.info(f"Stage : {STAGE_NAME} completed...")
    except Exception as e:
        raise e
