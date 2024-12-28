import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Exemple de données synthétiques
data = {
    'X': np.arange(1, 101),  # Feature (de 1 à 100)
    'Y': np.arange(1, 101) * 2 + np.random.randn(100) * 10  # Target avec du bruit
}
df = pd.DataFrame(data)

# Affichage des données
print(df.head())
plt.scatter(df['X'], df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Données Synthétiques')
plt.show()

# Séparer les données en X (features) et Y (target)
X = df[['X']]  # Nécessite un format 2D pour scikit-learn
Y = df['Y']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Entraînement : {len(X_train)} exemples, Test : {len(X_test)} exemples")

# Créer le modèle
model = LinearRegression()

# Entraîner le modèle sur les données d'entraînement
model.fit(X_train, Y_train)

# Afficher les coefficients du modèle
print(f"Coefficient : {model.coef_[0]}, Intercept : {model.intercept_}")

# Faire des prédictions sur l'ensemble de test
Y_pred = model.predict(X_test)

# Calculer l'erreur quadratique moyenne (MSE)
mse = mean_squared_error(Y_test, Y_pred)
print(f"Erreur Quadratique Moyenne : {mse:.2f}")

# Afficher les prédictions par rapport aux valeurs réelles
plt.scatter(X_test, Y_test, color='blue', label='Vraies valeurs')
plt.scatter(X_test, Y_pred, color='red', label='Prédictions')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Prédictions vs Réalité')
plt.show()
