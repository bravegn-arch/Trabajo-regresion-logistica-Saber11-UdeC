import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 42

def entrenar_modelo():
    print("[3/4] Entrenando el modelo de Regresión Logística...")
    clean_path = os.path.join(".", "data", "clean_saber11_clasificacion.csv")

    if not os.path.exists(clean_path):
        raise FileNotFoundError("❌ Archivo clean_saber11_clasificacion.csv no encontrado.")

    df_clean = pd.read_csv(clean_path)
    X = df_clean[['punt_lectura_critica', 'punt_matematicas', 'punt_ingles', 'estrato']]
    y = df_clean['alto_rendimiento']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)

    # Escalado de características (recomendado para Regresión Logística)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertir de nuevo a DataFrame para mantener nombres de columnas
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    modelo = LogisticRegression(random_state=SEED)
    modelo.fit(X_train_scaled, y_train)

    print(f"✔ Modelo entrenado. Intercepto (Beta_0): {modelo.intercept_[0]:.4f}")
    for col, coef in zip(X.columns, modelo.coef_[0]):
        print(f"  - Coeficiente ({col}): {coef:.4f}")

    return modelo, X_train_scaled, X_test_scaled, y_train, y_test, df_clean

if __name__ == "__main__":
    entrenar_modelo()
