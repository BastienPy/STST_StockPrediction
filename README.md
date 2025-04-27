# STST Stock Prediction

## Description

Ce projet implémente un modèle de prédiction de séries temporelles spatio-temporelles pour les données boursières. Il utilise des techniques avancées comme l'encodage temporel avec Date2Vec et des fenêtres glissantes pour capturer les dépendances temporelles et spatiales.

## Fonctionnalités

- Chargement et prétraitement des données boursières.
- Encodage temporel basé sur le modèle Date2Vec.
- Création de fenêtres glissantes pour les séries temporelles.
- Modèle d'encodage spatio-temporel pour les prédictions.
- Support pour la mise en cache des données prétraitées.
- Division des données en ensembles d'entraînement, de validation et de test.

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone <URL_DU_DEPOT>
   cd STST_StockPrediction
   ```

2. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### Entraînement du modèle

1. Configurez les paramètres dans le script principal stst.py.
2. Lancez l'entraînement :
   ```bash
   python stst.py
   ```

### Encodage temporel avec Date2Vec

Le modèle Date2Vec est utilisé pour générer des embeddings temporels. Vous pouvez utiliser un modèle pré-entraîné ou entraîner un nouveau modèle en suivant les instructions dans le dossier `Date2Vec`.

## Structure du projet

```
STST_StockPrediction/
├── Date2Vec/               # Implémentation de Date2Vec
├── data/                   # Données brutes et prétraitées
├── models/                 # Modèles sauvegardés
├── stst.py                 # Script principal
├── dataset.py              # Gestion des données et des fenêtres glissantes
├── README.md               # Documentation
└── requirements.txt        # Dépendances Python
```

## Résultats

Les résultats de l'entraînement, y compris les courbes de perte et d'exactitude, sont sauvegardés sous forme de graphiques dans le dossier de sortie.
