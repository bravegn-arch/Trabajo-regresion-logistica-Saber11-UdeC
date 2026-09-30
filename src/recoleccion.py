import os
import requests
import pandas as pd

def recolectar_datos_api(limit=5000):
    print(f"[1/4] Consultando la API de Datos Abiertos Colombia (Límite: {limit} registros)...")
    url = f"https://www.datos.gov.co/resource/rnvb-vnyh.json?$limit={limit}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        
        raw_dir = os.path.join(".", "data")
        os.makedirs(raw_dir, exist_ok=True)
        raw_path = os.path.join(raw_dir, "raw_saber11.csv")
        df.to_csv(raw_path, index=False)
        print(f"✔ Datos descargados y guardados en: {raw_path}")
        return df
    else:
        raise Exception(f"❌ Error al consultar la API: {response.status_code}")

if __name__ == "__main__":
    recolectar_datos_api()
