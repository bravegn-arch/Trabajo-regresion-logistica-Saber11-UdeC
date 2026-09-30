import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

def evaluar_y_graficar(modelo, X_train, X_test, y_train, y_test, df_clean):
    fig_dir = os.path.join(".", "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    # MÉTICAS
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n================ MÉTRICAS DE CLASIFICACIÓN ================")
    print(f"  - Exactitud (Accuracy):  {acc:.4f}")
    print(f"  - Precisión (Precision): {prec:.4f}")
    print(f"  - Sensibilidad (Recall): {rec:.4f}")
    print(f"  - F1-Score:              {f1:.4f}")
    print(f"  - Área Bajo Curva ROC:   {auc:.4f}")
    print("===========================================================\n")

    # 1. Matriz de Confusión
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Bajo (<250)', 'Alto (>=250)'],
                yticklabels=['Bajo (<250)', 'Alto (>=250)'])
    plt.title('Matriz de Confusión', fontsize=12)
    plt.xlabel('Predicción', fontsize=10)
    plt.ylabel('Valor Real', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "01_matriz_conexion.png"), dpi=300)
    plt.close()

    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#38688a', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=1.5, label='Clasificador Aleatorio')
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=10)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=10)
    plt.title('Curva ROC (Receiver Operating Characteristic)', fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_curva_roc.png"), dpi=300)
    plt.close()

    print("✔ Gráficas de clasificación guardadas en 'reports/figures/'.")
