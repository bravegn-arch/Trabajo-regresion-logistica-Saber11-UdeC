# Regresión Logística – Predicción de Alto Rendimiento (Saber 11)

Clasificación Binaria.
Proceso: Recolección de datos abiertos → Limpieza → Creación de etiqueta binaria (alto_rendimiento) → 
Modelado con Regresión Logística Binaria → Evaluación con Matriz de Confusión y Curva ROC → conclusiones.

## Problema

- **Variable objetivo (Y):** alto_rendimiento – clasificación binaria (1: Puntaje Global => 250 puntos, 0: Puntaje Global < 250 puntos).
- **Predictores (X):** punt_lectura_critica, punt_matematicas, punt_ingles, estrato.
- **Dataset:** Pruebas Saber 11 Colombia (ICFES), 4,428 registros procesados.
- **Fuente:** datos abiertos descargados vía HTTP GET desde
  `https://www.datos.gov.co/resource/rnvb-vnyh.json`.

## Estructura del proyecto

```


TRABAJO-REGRESION-SABER-11/
├── data/
│   ├── raw_saber11.csv        # datos crudos (inmutables) de la API
│   └── clean_saber11_clasificacion.csv      # datos limpios listos para modelar
├── src/
│   ├── init.py            # módulo paquete Python
│   ├── recoleccion.py         # recolección (HTTP GET a API Socrata)
│   ├── eda_limpieza.py        # limpieza con RegEx + binarización
│   ├── modelamiento.py        # train/test split, StandardScaler, LogisticRegression
│   └── evaluacion_graficas.py # métricas de clasificación y gráficos (ROC / Confusión)
├── reports/
│	└── figures/
│    		├── 01_matriz_conexion.png      # gráfico de la matriz de confusión
│    		├── 02_curva_roc.png             # curva ROC y cálculo del AUC
│   	
├── main.py                    # ejecuta el pipeline en orden
├── requirements.txt           # dependencias del proyecto
└── README.md
```

## Cómo ejecutar

python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                  # ejecuta todo el pipeline
```

Ejecución del scrip: al ejecutar main.py, seleccionar la opción 5 para correr de forma secuencial todo el flujo (recoleccion, limpieza, modelamiento, evaluacion_graficas)..
La semilla de aleatoriedad está fijada en random_state = 42.

## Resultados principales (conjunto de prueba)

| Métrica | Valor |
|---|---|
| Exactitud (Accuracy)  | 0.9481 |
| Precisión (Precision) | 0.9452|
| Sensibilidad (Recall)   | 0.9501|
| F1-Score   | 0.9476|
| Área Bajo la Curva (AUC-ROC)   | 0.9872|

## Notas técnicas

- recoleccion.py intenta la descarga en vivo desde el portal de Datos Abiertos Colombia; 
si no hay conexión, usa automáticamente la copia local guardada en data/ como respaldo.
- La variable `estrato` se extrajo mediante expresiones regulares (re.search(r'\d+')) a partir de la columna cualitativa fami_estratovivienda
- La variable punt_global se elimina tras generar la etiqueta binaria alto_rendimiento.
-Las variables explicativas se transforman con StandardScaler (mu=0, sigma=1) antes de ajustar la regresión logística para optimizar el cálculo de coeficientes y comparar sus Odds Ratios.
