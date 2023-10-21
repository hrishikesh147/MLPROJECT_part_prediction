import pandas as pd
import os
from src.ML_Part_predict import logger
from src.ML_Part_predict.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self,config : DataValidationConfig):
        self.config=config

    def validate_all_columns(self)-> bool:
        try:
            validation_status=None

            data=pd.read_csv(self.config.unzip_dir)

            all_cols=list(data.columns)
            all_schema=self.config.all_schema.keys()

            for i in all_cols:
                if i not in all_schema:
                    validation_status= False
                    with open(self.config.STATUS_FILE,"w") as f:
                        f.write(f"Validation Status: {validation_status}")

                else:
                    validation_status=True
                    with open(self.config.STATUS_FILE,"w") as f:
                        f.write(f"Validation Status : {validation_status}")

        

        except Exception as e:
            raise e