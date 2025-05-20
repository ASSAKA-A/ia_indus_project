# Projet de Prédiction des Sinistres en Assurance
## Une approche par Machine Learning

---

### 1. Introduction

#### Objectif du Projet
- Prédiction de la fréquence des sinistres en assurance
- Estimation du coût moyen des sinistres
- Calcul du coût total prévisionnel

#### Technologies Utilisées
- Python 3.9+
- FastAPI pour l'API REST
- XGBoost pour les modèles de ML
- Docker pour la conteneurisation
- Tests automatisés avec pytest

---

### 2. Architecture du Projet

#### Structure
```
.
├── app.py              # API FastAPI
├── preprocessing.py    # Prétraitement
├── train_xgboost.py   # Entraînement
├── predict.py         # Prédictions
└── test_api.py        # Tests
```

#### Composants Principaux
- API REST
- Pipeline de prétraitement
- Modèles XGBoost
- Suite de tests
- Conteneurisation Docker

---

### 3. Fonctionnalités Clés

#### API Endpoints
1. `/predict_freq`
   - Prédiction de la fréquence des sinistres
   - Retourne une valeur entre 0 et 1

2. `/predict_cm`
   - Prédiction du coût moyen
   - Estimation en euros

3. `/predict_sinistre`
   - Prédiction complète (fréquence + coût)
   - Calcul du coût total prévisionnel

---

### 4. Modèle de Machine Learning

#### Caractéristiques du Modèle
- Algorithme : XGBoost
- Variables d'entrée : +200 features
- Double prédiction :
  - Modèle de fréquence
  - Modèle de coût

#### Features Importantes
- Données géographiques
- Informations météorologiques
- Caractéristiques des bâtiments
- Données démographiques
- Historique des sinistres

---

### 5. Pipeline de Données

#### Prétraitement
1. Nettoyage des données
2. Gestion des valeurs manquantes
3. Encodage des variables catégorielles
4. Normalisation des features

#### Validation
- Tests unitaires
- Validation croisée
- Métriques de performance

---

### 6. Déploiement

#### Infrastructure
- Conteneurisation Docker
- CI/CD avec GitHub Actions
- Tests automatisés

#### Monitoring
- Endpoints de santé
- Logs des prédictions
- Métriques de performance

---

### 7. Performance et Métriques

#### Métriques Clés
- Temps de réponse API < 200ms
- Précision des prédictions
- Couverture des tests > 80%

#### Scalabilité
- Architecture REST
- Containerisation
- Optimisation des performances

---

### 8. Sécurité et Conformité

#### Mesures de Sécurité
- Validation des données d'entrée
- Gestion des erreurs
- Logs sécurisés

#### Conformité
- Documentation API
- Tests de régression
- Versioning des modèles

---

### 9. Perspectives

#### Améliorations Futures
- Interface utilisateur web
- Optimisation continue des modèles
- Ajout de nouvelles features
- Analyse des erreurs de prédiction

#### Maintenance
- Mise à jour des dépendances
- Monitoring continu
- Formation continue des modèles

---

### 10. Conclusion

#### Points Forts
- API REST moderne et performante
- Modèles ML robustes
- Tests complets
- Architecture scalable

#### Contact
- Documentation : `/docs`
- Tests : `pytest`
- Déploiement : `docker-compose up`

---

### Merci de votre attention !

#### Questions ? 