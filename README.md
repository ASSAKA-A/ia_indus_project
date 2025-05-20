# Projet de Prédiction de la Fréquence des Sinistres

Ce projet est une application de machine learning déployable qui permet de prédire la fréquence des sinistres en assurance. Il utilise XGBoost comme modèle principal et expose une API REST pour les prédictions en temps réel.

## Structure du Projet

```
.
├── app.py              # Application FastAPI
├── config.py           # Configuration et paramètres
├── preprocessing.py    # Prétraitement des données
├── train_xgboost.py   # Entraînement du modèle XGBoost
├── predict.py         # Module de prédiction
├── test_api.py        # Tests de l'API
├── Dockerfile         # Configuration Docker
├── requirements.txt   # Dépendances du projet
└── data/             # Dossier des données
    └── ...
└── models/           # Dossier des modèles sauvegardés
    └── ...
```

## Prérequis

- Python 3.8+
- pip
- virtualenv (recommandé)

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd [NOM_DU_PROJET]
```

2. Créer et activer un environnement virtuel (recommandé) :
```bash
python -m venv env
source env/bin/activate  # Sur Linux/Mac
# ou
.\env\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Configuration

Le fichier `config.py` contient tous les paramètres de configuration :
- Chemins des données
- Paramètres du modèle
- Configuration de l'API
- Paramètres de prétraitement

## Prétraitement des Données

Le module `preprocessing.py` gère :
- Le nettoyage des données
- La gestion des valeurs manquantes
- L'encodage des variables catégorielles
- La normalisation des features
- La validation des données

## Entraînement du Modèle

Pour entraîner un nouveau modèle :
```bash
python train_xgboost.py
```

Le script :
1. Charge et prétraite les données
2. Optimise les hyperparamètres avec Optuna
3. Entraîne le modèle XGBoost
4. Évalue les performances
5. Sauvegarde le modèle et les métriques

## API REST

L'API est développée avec FastAPI et expose les endpoints suivants :

1. Démarrer l'API :
```bash
uvicorn app:app --reload
```

2. Endpoints :
- `POST /predict` : Prédiction de la fréquence des sinistres
- `GET /health` : Vérification de la santé de l'API
- `GET /model-info` : Informations sur le modèle actuel

Documentation interactive disponible sur : `http://localhost:8000/docs`

## Tests

Le projet inclut des tests unitaires et d'intégration :
```bash
pytest
```

Pour la couverture des tests :
```bash
pytest --cov
```

## Docker

Construction de l'image :
```bash
docker build -t prediction-sinistres .
```

Lancement du conteneur :
```bash
docker run -p 8000:8000 prediction-sinistres
```

## Format des Données

### Entrée API
```json
{
    "data": [{
        "feature1": "valeur1",
        "feature2": "valeur2",
        ...
    }]
}
```

### Sortie API
```json
{
    "predictions": [0.123],
    "model_version": "1.0.0",
    "timestamp": "2024-02-20T10:00:00Z"
}
```

## Dépendances Principales

- fastapi==0.109.2
- uvicorn==0.27.1
- pandas==2.2.0
- numpy==1.26.3
- scikit-learn==1.4.0
- xgboost==2.0.3
- pydantic==2.6.1
- optuna
- pytest
- pytest-cov
