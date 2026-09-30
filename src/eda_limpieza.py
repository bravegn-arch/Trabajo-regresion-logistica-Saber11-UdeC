import os
import re
import pandas as pd
import numpy as np

def limpiar_y_explorar_datos():
    print("[2/4] Ejecutando limpieza de datos y creación de variable binaria...")
    raw_path = os.path.join(".", "data", "raw_saber11.csv")
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError("❌ Archivo raw_saber11.csv no encontrado.")
        
    df = pd.read_csv(raw_path)
    cols = ['punt_global', 'punt_lectura_critica', 'punt_matematicas', 'punt_ingles', 'fami_estratovivienda']
    
    # Filtrar columnas existentes
    cols_existentes = [c for c in cols if c in df.columns]
    df = df[cols_existentes].dropna()

    # Limpiar estrato
    def clean_estrato(text):
        match = re.search(r'\d+', str(text))
        return float(match.group()) if match else np.nan

    if 'fami_estratovivienda' in df.columns:
        df['estrato'] = df['fami_estratovivienda'].apply(clean_estrato)
        df = df.drop(columns=['fami_estratovivienda'])

    # Convertir a numérico
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()

    # CREACIÓN DE LA VARIABLE OBJETIVO BINARIA (0 o 1)
    # 1 si punt_global >= 250, 0 en caso contrario
    df['alto_rendimiento'] = (df['punt_global'] >= 250).astype(int)
    
    # Eliminamos el punt_global original para evitar fuga de datos (Data Leakage)
    df_clean = df.drop(columns=['punt_global'])

    clean_path = os.path.join(".", "data", "clean_saber11_clasificacion.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"✔ Dataset procesado guardado en: {clean_path}")
    print(f"  - Balance de clases: {df_clean['alto_rendimiento'].value_counts().to_dict()}")
    return df_clean

if __name__ == "__main__":
    limpiar_y_explorar_datos()
