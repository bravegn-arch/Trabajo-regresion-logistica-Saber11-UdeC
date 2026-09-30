import sys
import os

# Asegurar que src esté en el path de Python
sys.path.append(os.path.abspath("src"))

from src import recoleccion
from src import eda_limpieza
from src import modelamiento
from src import evaluacion_graficas

def mostrar_menu():
    print("\n============================================================")
    print(" PIPELINE DE REGRESIÓN LOGÍSTICA - SABER 11 COLOMBIA")
    print(" Autor: Brayan Yesid Vega Niño")
    print("============================================================")
    print(" 1. Recolectar datos desde API Socrata")
    print(" 2. Limpieza y Creación de Clase Binaria (alto_rendimiento)")
    print(" 3. Entrenar Modelo de Regresión Logística")
    print(" 4. Evaluar y Generar Gráficas (Matriz de Confusión y ROC)")
    print(" 5. EJECUTAR PIPELINE COMPLETO (Pasos 1 a 4)")
    print(" 0. Salir")
    print("============================================================")

def ejecutar_pipeline_completo():
    print("\n>>> INICIANDO PIPELINE DE REGRESIÓN LOGÍSTICA <<<\n")
    recoleccion.recolectar_datos_api(limit=5000)
    df_clean = eda_limpieza.limpiar_y_explorar_datos()
    modelo, X_train, X_test, y_train, y_test, df_clean = modelamiento.entrenar_modelo()
    evaluacion_graficas.evaluar_y_graficar(modelo, X_train, X_test, y_train, y_test, df_clean)
    print("\n✔ PIPELINE EJECUTADO CON ÉXITO.")

def main():
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción (0-5): ").strip()
        if opcion == "1":
            recoleccion.recolectar_datos_api()
        elif opcion == "2":
            eda_limpieza.limpiar_y_explorar_datos()
        elif opcion == "3":
            modelamiento.entrenar_modelo()
        elif opcion == "4":
            df_clean = eda_limpieza.limpiar_y_explorar_datos()
            modelo, X_train, X_test, y_train, y_test, df_clean = modelamiento.entrenar_modelo()
            evaluacion_graficas.evaluar_y_graficar(modelo, X_train, X_test, y_train, y_test, df_clean)
        elif opcion == "5":
            ejecutar_pipeline_completo()
        elif opcion == "0":
            break
        else:
            print("❌ Opción inválida. Intenta nuevamente.")

if __name__ == "__main__":
    main()
