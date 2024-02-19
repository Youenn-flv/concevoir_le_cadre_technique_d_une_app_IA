# Import des librairies uvicorn, pickle, FastAPI, File, UploadFile, BaseModel
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from pydantic import BaseModel
import pickle
import pandas as pd

import mlflow as ml
import os
import boto3 as bo



# Création des tags
tags = [
       {
              "name": "Hello name V1",
              "description": "te répond Hello World pour vérifier la conexion à l'api",
       },
       {
              "name": "Predict V1",
              "description": "Prediction du modelle 1",
       },
       {
              "name": "Predict V2",
              "description": "Prediction du modelle 2",
       },
]

# Création de l'application
app = FastAPI(
       title="API de prediction",
       description= "Predictions",
       version= "1.0.0",
       openapi_tags= tags
)

# Point de terminaison avec paramètre
@app.get("/hello", tags=["Hello name V1"])
def hello(name: str='World'):
        return {"message": f"Hello {name}"}



# Création du modèle de données pour le modéle 1 ('Gender', 'Age', 'Physical Activity Level', 'Heart Rate', 'Daily Steps', 'BloodPressure_high', 'BloodPressure_low', 'Sleep Disorder'])
class Credit(BaseModel):
        Gender : int
        Age : int
        Physical_Activity_Level : int
        Heart_Rate : int
        Daily_Steps : int
        BloodPressure_high : int
        BloodPressure_low : int        ## !!! atention c'est probablement pas le bon typage

# Charger le modèle 1
with open('a_1_19-02-24_concevoir_le_cadre_technique_d_une_app_IA/model_1.pkl', 'rb') as file:
    model_1 = pickle.load(file)


# Point de terminaison : Prédiction 1
@app.post("/predict", tags=["Predict V1"])
def predict(credit: Credit) :
    # Transformation des données dans le bon format pour la prédiction
    data = [[credit.Gender,
             credit.Age,
             credit.Physical_Activity_Level,
             credit.Heart_Rate,
             credit.Daily_Steps,
             credit.BloodPressure_high,
             credit.BloodPressure_low]]
    
    # Prédiction avec le modèle
    prediction = model_1.predict(data)
    
    return {"prediction": prediction[0]}  # Supposant que le modèle renvoie

# Création du modèle de données pour le modéle 2 ('Physical Activity Level', 'Heart Rate', 'Daily Steps', 'Sleep Disorder')
class HealthData(BaseModel):
    Physical_Activity_Level: int
    Heart_Rate: int
    Daily_Steps: int
    Sleep_Disorder: int

# Charger le modèle 2
with open('a_1_19-02-24_concevoir_le_cadre_technique_d_une_app_IA/model_2.pkl', 'rb') as file:
    model_2 = pickle.load(file)


# Point de terminaison : Prédiction 2
@app.post("/predict2", tags=["Predict V2"])
def predict2(data: HealthData):
    prediction = model_2.predict(data)
    return {"prediction": prediction}


# Démarage de l'application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

