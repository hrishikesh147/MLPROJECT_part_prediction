from src.ML_Part_predict.constants import *
from src.ML_Part_predict.utils.common import read_yaml,create_directories
from src.ML_Part_predict.entity.config_entity import (DataIngestionConfig,DataValidationConfig,DataTransformConfig,ModelTrainerConfig,ModelEvaluationConfig)


class ConfigurationManager:
    def __init__(self,config_filepath=CONFIG_FILE_PATH,params_filepath=PARAMS_FILE_PATH,schema_filepath=SCHEMA_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir= config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config=self.config.data_validation
        schema=self.schema.COLUMNS

        create_directories([config.root_dir])

        data_val_config = DataValidationConfig(
            root_dir=config.root_dir,
            unzip_dir=config.unzip_dir,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema
        )

        return data_val_config
    

    def get_data_transformation_config(self) -> DataTransformConfig:
        config=self.config.data_transformation

        create_directories([config.root_dir])

        data_transform_config=DataTransformConfig(
            root_dir=config.root_dir,
            data_dir=config.data_dir
        )
        return data_transform_config
    
    def get_model_trainer_config(self)->ModelTrainerConfig:
        config=self.config.model_trainer
        params=self.params.random_forest
        schema=self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config=ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            target_column=schema.name,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            min_samples_leaf=params.min_samples_leaf,
            bootstrap=params.bootstrap

        )

        return model_trainer_config
    
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config=self.config.model_evaluation
        params=self.params.random_forest  #
        schema=self.schema.TARGET_COLUMN

        create_directories([config.root_dir])   

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = config.root_dir,
            test_data_path = config.test_data_path,
            model_path = config.model_path,
            all_params = params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            mlflow_uri = "https://dagshub.com/hrishikeshbhagawati01/MLPROJECT_part_prediction.mlflow"

        )

        return model_evaluation_config