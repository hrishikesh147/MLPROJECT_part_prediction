from flask import Flask,render_template,request
import os
import pandas as pd
import numpy as np
from src.ML_Part_predict.pipeline.PredictionPipeleine import PredictionPipeline

app=Flask(__name__)

@app.route('/',methods=['GET'])
def homaPage():
    return render_template("index.html")

@app.route('/train',methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successfull"

def process_input_data(data):
    # Convert the list to DataFrame
    df = pd.DataFrame([data], columns=[
        'VehicleModel', 'VehicleYear', 'PartName', 'HoursOfOperation',
        'SupplierName', 'ClaimDate', 'SettlementAmount',
        'MaintenanceFrequency', 'EnvironmentCondition', 'OperationalIntensity',
        'WarrantyStatus', 'PreviousFailures', 'pass_fail'
    ])

    # Apply the transformations you did during training
    df['VehicleYear'] = df['VehicleYear'].astype('str')
    df['ClaimYear'] = df['ClaimDate'].str.split('-').str[0]
    df['MaintenanceFrequency'] = df['MaintenanceFrequency'].str.replace('hours', '').astype('Int64')
    df['PreviousFailures'] = df['PreviousFailures'].astype('Int64')

    # One-hot encoding. Adjust according to the exact transformations you used during training.
    df = pd.get_dummies(df, columns=[
        'VehicleModel', 'VehicleYear', 'PartName', 'SupplierName',
        'EnvironmentCondition', 'OperationalIntensity', 'WarrantyStatus',
        'ClaimYear'
    ], drop_first=True)

    df.drop(['ClaimDate'], axis=1, inplace=True)

    return df


@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # reading the inputs given by the user
            VehicleModel = request.form['VehicleModel']
            VehicleYear = int(request.form['VehicleYear'])
            PartName = request.form['PartName']
            HoursOfOperation = float(request.form['HoursOfOperation'])
            SupplierName = request.form['SupplierName']
            ClaimDate = request.form['ClaimDate']  # Might need datetime conversion
            SettlementAmount = float(request.form['SettlementAmount'])
            MaintenanceFrequency = request.form['MaintenanceFrequency']
            EnvironmentCondition = request.form['EnvironmentCondition']
            OperationalIntensity = request.form['OperationalIntensity']
            WarrantyStatus = request.form['WarrantyStatus']
            PreviousFailures = int(request.form['PreviousFailures'])
            pass_fail = int(request.form['pass_fail'])

            data = [
                VehicleModel, VehicleYear, PartName, HoursOfOperation,
                SupplierName, ClaimDate, SettlementAmount,
                MaintenanceFrequency, EnvironmentCondition, OperationalIntensity,
                WarrantyStatus, PreviousFailures, pass_fail
            ]

            processed_data = process_input_data(data)

            obj = PredictionPipeline()
            predict = obj.predict(processed_data)

            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong: ' + str(e)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",port = 8080)
