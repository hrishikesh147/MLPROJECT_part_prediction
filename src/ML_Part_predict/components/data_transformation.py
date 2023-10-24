from src.ML_Part_predict import logger
import pandas as pd
import os
from src.ML_Part_predict import logger
from sklearn.model_selection import train_test_split
from src.ML_Part_predict.entity.config_entity import DataTransformConfig
from pathlib import Path


class FeatureEngineering:
    def __init__(self,config: DataTransformConfig):
        logger.info("Feature Engineering Started")
        self.config=config

    def Random_Sample_imputation(self,data,feature):
        random_sample=data[feature].dropna().sample(data[feature].isnull().sum())               
        random_sample.index=data[data[feature].isnull()].index
        data.loc[data[feature].isnull(),feature]=random_sample

    def handling_outliers(self,datas,columns):
        for i in columns:
            q1=datas[i].quantile(0.25)
            q3=datas[i].quantile(0.75)
            IQR=q3-q1
            upper_lim=q3+1.5*IQR
            lower_lim=q1-1.5*IQR
            datas[i]=datas[i].apply(lambda x:upper_lim if x>upper_lim else lower_lim if x<lower_lim else x)
        
        return datas

    def feature_E(self):
        try:
            df=pd.read_csv(self.config.data_dir)
            logger.info(f"the columns are {df.columns} and shape is : {df.shape}")

            df.drop_duplicates(inplace=True)
            logger.info(f"duplicated data number: {df.duplicated().sum()}. Duplicates deletion completed")

            df['VehicleYear']=df['VehicleYear'].astype('str')
            logger.info(f"vehicleYear dtype changed to object")
            df['ClaimYear']=df['ClaimDate'].str.split('-').str[0]
            logger.info(f"feature ClaimYear extracted from ClaimDate")
            df['MaintenanceFrequency']=df['MaintenanceFrequency'].str.replace('hours','').astype('Int64')
            logger.info(f"in feature MaintenanceFrequency hours is removed and dtype changed to Int64")
            df['PreviousFailures']=df['PreviousFailures'].astype('Int64')
            logger.info(f"in feature PreviousFailures dtype changed to Int64")

            for col in df.columns:
                self.Random_Sample_imputation(data=df, feature=col)
            logger.info(f"replacing null values with random imputation completed...")

            self.handling_outliers(datas=df,columns=['HoursOfOperation','SettlementAmount','MaintenanceFrequency','PreviousFailures'])
            logger.info(f"Outlier handling completed...")

            df['WarrantyStatus'].replace({'Out of Warranty':'out_of_warranty','In Warranty':'in_warranty'},inplace=True)
            logger.info(f"warranty status values renamed as {df['WarrantyStatus'].unique()}")
            
            df=pd.get_dummies(df, columns=['VehicleModel','VehicleYear','PartName','SupplierName','EnvironmentCondition','OperationalIntensity','WarrantyStatus','ClaimYear'], drop_first=True)
            df['pass_fail'].replace({'pass': 1, 'fail': 0}, inplace=True)    
            logger.info(f"implementation of pd.dummies completed... ")

            df.drop(['ClaimDate'],axis=1,inplace=True)
            logger.info(f"features dropped are : ClaimDate ")

            save_path = Path("artifacts/data_transformation/FE_data.csv")
            logger.info(f"Saving file to {save_path}")
            df.to_csv(save_path, index=False)

            logger.info(f"Feature Engineering Completed and file saved Successfully")
    
        except Exception as e:
            raise e
        

class DataTransformation:
    def __init__(self,config: DataTransformConfig):
        logger.info(f"Data Transformation initiated")
        self.config=config

    def train_test_split(self):
        try:
            logger.info("reading the Feature Engineered csv file")
            df=pd.read_csv(os.path.join(self.config.root_dir,"FE_data.csv"))
            logger.info("reading the Feature Engineered csv file completed...")

            train,test=train_test_split(df)
            logger.info("Splitting of train test done...")

            train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index=False)
            test.to_csv(os.path.join(self.config.root_dir,"test.csv"),index=False)

            logger.info(train.shape)
            logger.info(test.shape)
        
        except Exception as e:
            raise e
        
