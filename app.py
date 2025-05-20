"""
API de prédiction de sinistres d'assurance.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import time
from preprocessing import preprocess_test

# Modèles de requête/réponse
class PredictionRequest(BaseModel):
    """
    Format de la requête pour les prédictions.
    
    Attributes:
        data: Liste des caractéristiques du contrat d'assurance
    """
    data: List[Dict[str, Any]]

class FreqResponse(BaseModel):
    """
    Réponse pour la prédiction de fréquence.
    
    Attributes:
        freq: Probabilité de sinistre (entre 0 et 1)
        prediction_time: Temps de calcul en secondes
    """
    freq: float
    prediction_time: float

class CmResponse(BaseModel):
    """
    Réponse pour la prédiction de coût.
    
    Attributes:
        cm: Coût moyen prédit du sinistre
        prediction_time: Temps de calcul en secondes
    """
    cm: float
    prediction_time: float

class SinistreResponse(BaseModel):
    """
    Réponse complète de prédiction.
    
    Attributes:
        freq: Probabilité de sinistre (entre 0 et 1)
        cm: Coût moyen prédit
        total_cost: Coût total prédit (freq * cm)
        prediction_time: Temps de calcul en secondes
    """
    freq: float
    cm: float
    total_cost: float
    prediction_time: float

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Prédiction Sinistres",
    description="""
    API de prédiction de sinistres d'assurance utilisant des modèles XGBoost.

    Points d'accès disponibles:
    /predict_freq - Prédit la probabilité d'occurrence d'un sinistre
    /predict_cm - Prédit le coût moyen d'un sinistre
    /predict_sinistre - Fournit la prédiction complète (fréquence, coût et coût total)

    Toutes les requêtes doivent contenir un objet JSON avec un champ 'data' contenant les caractéristiques du contrat.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Chargement des modèles
try:
    freq_data = joblib.load('models/xgb_freq_model.joblib')
    cm_data = joblib.load('models/xgb_cm_model.joblib')
    freq_model = freq_data['model']
    cm_model = cm_data['model']
except Exception as e:
    print(f"Erreur de chargement des modèles: {str(e)}")
    raise

@app.get("/health")
async def health_check():
    """
    Vérifie l'état de l'API et des modèles.
    
    Returns:
        dict: État de santé de l'API et des modèles
    """
    return {
        "status": "healthy",
        "models_loaded": True,
        "freq_model_type": str(type(freq_model)),
        "cm_model_type": str(type(cm_model))
    }

@app.post("/predict_freq", response_model=FreqResponse)
async def predict_freq(request: PredictionRequest):
    """
    Prédit la probabilité d'occurrence d'un sinistre.
    
    Args:
        request: Caractéristiques du contrat d'assurance
        
    Returns:
        FreqResponse: Probabilité prédite et temps de calcul
    """
    try:
        start_time = time.time()
        
        df = pd.DataFrame(request.data)
        df_prep = preprocess_test(df)
        freq_pred = freq_model.predict(df_prep)
        freq_pred = np.clip(freq_pred, 0, None)
        
        prediction_time = time.time() - start_time
        
        return FreqResponse(
            freq=float(freq_pred[0]),
            prediction_time=prediction_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction: {str(e)}"
        )

@app.post("/predict_cm", response_model=CmResponse)
async def predict_cm(request: PredictionRequest):
    """
    Prédit le coût moyen d'un sinistre.
    
    Args:
        request: Caractéristiques du contrat d'assurance
        
    Returns:
        CmResponse: Coût prédit et temps de calcul
    """
    try:
        start_time = time.time()
        
        df = pd.DataFrame(request.data)
        df_prep = preprocess_test(df)
        cm_pred = cm_model.predict(df_prep)
        cm_pred = np.clip(cm_pred, 0, None)
        
        prediction_time = time.time() - start_time
        
        return CmResponse(
            cm=float(cm_pred[0]),
            prediction_time=prediction_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction: {str(e)}"
        )

@app.post("/predict_sinistre", response_model=SinistreResponse)
async def predict_sinistre(request: PredictionRequest):
    """
    Fournit une prédiction complète du sinistre.
    
    Args:
        request: Caractéristiques du contrat d'assurance
        
    Returns:
        SinistreResponse: Fréquence, coût et coût total prédits
    """
    try:
        start_time = time.time()
        
        df = pd.DataFrame(request.data)
        df_prep = preprocess_test(df)
        
        freq_pred = freq_model.predict(df_prep)
        cm_pred = cm_model.predict(df_prep)
        
        freq_pred = np.clip(freq_pred, 0, None)
        cm_pred = np.clip(cm_pred, 0, None)
        
        total_cost = float(freq_pred[0] * cm_pred[0])
        prediction_time = time.time() - start_time
        
        return SinistreResponse(
            freq=float(freq_pred[0]),
            cm=float(cm_pred[0]),
            total_cost=total_cost,
            prediction_time=prediction_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de prédiction: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)