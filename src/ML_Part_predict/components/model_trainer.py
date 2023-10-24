import pandas as pd
import os
from src.ML_Part_predict import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score,precision_score,recall_score)
from src.ML_Part_predict.entity.config_entity import ModelTrainerConfig
import joblib

class ModelTrainer:
    def __init__(self,config:ModelTrainerConfig):
        self.config=config

    def train(self):
        train_data=pd.read_csv(self.config.train_data_path)
        test_data=pd.read_csv(self.config.test_data_path)
        logger.info("reading of train data and test data successfull")

        logger.info("X_train,X_test,y_train,y_test split initiated")
        X_train=train_data.drop([self.config.target_column],axis=1)
        y_train=train_data[[self.config.target_column]]
        X_test=test_data.drop([self.config.target_column],axis=1)
        y_test=test_data[[self.config.target_column]]
        logger.info("X_train,X_test,y_train,y_test split successfull")

        random_grid = {'n_estimators': self.config.n_estimators,
               #'max_features': self.config.ma,
               'max_depth': self.config.max_depth,
               'min_samples_split': self.config.min_samples_split,
               'min_samples_leaf': self.config.min_samples_leaf,
               'bootstrap': self.config.bootstrap}
        
        rf_random_CV = RandomizedSearchCV(estimator = RandomForestClassifier(class_weight='balanced'), param_distributions = random_grid, 
                               n_iter = 100, cv = 3,verbose=2, random_state=42, 
                               n_jobs =-1,scoring='neg_mean_squared_error')
        
        rf_random_CV.fit(X_train,y_train)
        logger.info("Randomised search CV fir Completed...")

        best_rf_model = rf_random_CV.best_estimator_
        logger.info(f"Randomised search CV best estimators: {best_rf_model}")

        best_params = rf_random_CV.best_params_
        logger.info(f"Randomised search CV best params: {best_params}")

        best_rf_model.fit(X_train,y_train)
        logger.info(f"Random forest fitting with best estimator completed...")

        #y_pred_hyp=best_rf_model.predict(X_test)

        # Adjusting the threshold to improve recall
        threshold = 0.3  # This value might need fine-tuning
        probs = best_rf_model.predict_proba(X_test)
        y_pred_hyp = [1 if prob[1] > threshold else 0 for prob in probs]

        logger.info("Random Forest WITH Best Parameters")
        logger.info(f"recall:{precision_score(y_test,y_pred_hyp)}")
        logger.info(f"precision:{precision_score(y_test,y_pred_hyp)}")
        logger.info(f"accuracy:{accuracy_score(y_test,y_pred_hyp)}")

        joblib.dump(rf_random_CV, os.path.join(self.config.root_dir,self.config.model_name))
        logger.info("Model dump successfully")