# Modèle de Prédiction avec Python

Ce projet est une implémentation simple d'un modèle de régression linéaire en Python. Il utilise des données synthétiques pour prédire une variable cible (Y) à partir d'une caractéristique (X). Ce projet est conçu comme un point de départ pour comprendre les bases de la régression et de l'apprentissage automatique avec Python.

## Fonctionnalités
- Génération de données synthétiques avec du bruit aléatoire.
- Visualisation des données à l'aide de Matplotlib.
- Séparation des données en ensembles d'entraînement et de test.
- Construction, entraînement et évaluation d'un modèle de régression linéaire.
- Affichage des prédictions et des erreurs.

## Technologies utilisées
- **Python 3.10+**
- **Bibliothèques :**
  - `numpy` : manipulation des données numériques.
  - `pandas` : manipulation de données tabulaires.
  - `matplotlib` : visualisation des données.
  - `scikit-learn` : implémentation du modèle et des métriques.

## Structure du code
1. **Génération de données synthétiques** :
   - Création d'une variable `X` (1 à 100) et d'une cible `Y` (proportionnelle à `X` avec du bruit aléatoire).
   - Visualisation des données générées.

2. **Préparation des données** :
   - Séparation des données en variables d'entrée (`X`) et cible (`Y`).
   - Division des données en ensembles d'entraînement et de test.

3. **Création et entraînement du modèle** :
   - Utilisation de `LinearRegression` pour entraîner un modèle sur les données d'entraînement.
   - Extraction des coefficients et de l'interception.

4. **Évaluation du modèle** :
   - Prédictions sur l'ensemble de test.
   - Calcul de l'erreur quadratique moyenne (MSE).
   - Visualisation des prédictions par rapport aux valeurs réelles.

## Instructions

### Prérequis
- Python installé sur votre machine.
- Installer les bibliothèques nécessaires avec :
  ```bash
  pip install numpy pandas scikit-learn matplotlib
