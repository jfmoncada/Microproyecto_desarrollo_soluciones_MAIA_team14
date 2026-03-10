"""
Módulo de Predicción — Customer Subscription Model

Descripción:
Script encargado de cargar el modelo de Machine Learning previamente
entrenado y realizar predicciones sobre nuevos datos de clientes.
Permite estimar la probabilidad de que un cliente se suscriba a partir
de sus características de compra.

Objetivo:
Facilitar el uso del modelo entrenado para realizar inferencias
sobre nuevos registros de clientes.

Proceso:
1. Carga del modelo entrenado almacenado en formato .pkl.
2. Recepción de los datos de entrada en formato diccionario.
3. Conversión de los datos a un DataFrame de Pandas.
4. Cálculo de la probabilidad de suscripción mediante el modelo.
5. Generación de la predicción final (0 = No suscripción, 1 = Suscripción).
6. Manejo de errores durante el proceso de predicción.

Entrada:
Diccionario con las características del cliente.

Salida:
- Probabilidad de suscripción.
- Predicción final del modelo.

Resultados esperados:
Permitir que el modelo entrenado pueda ser utilizado fácilmente
en APIs, dashboards o aplicaciones de analítica predictiva.

"""

import joblib
import pandas as pd

# =====================================================
# CARGAR MODELO ENTRENADO
# =====================================================

MODEL_PATH = "models/subscription_model.pkl"
print("Cargando modelo entrenado...")
model = joblib.load(MODEL_PATH)
print("Modelo cargado correctamente")


# =====================================================
# FUNCIÓN DE PREDICCIÓN
# =====================================================

def predict_subscription(input_data: dict):
    """
    Recibe un diccionario con las características de un cliente
    y devuelve:

    - Probabilidad de suscripción
    - Predicción final (0 = No, 1 = Sí)
    """

    try:

        # Convertir entrada a DataFrame
        df_input = pd.DataFrame([input_data])

        print("\n Datos recibidos para predicción:")
        print(df_input)

        # Probabilidad
        prob = model.predict_proba(df_input)[0][1]

        # Predicción
        pred = model.predict(df_input)[0]

        print(f"\n Probabilidad de suscripción: {prob:.4f}")

        if pred == 1:
            print("Predicción final: CLIENTE SE SUSCRIBIRÁ")
        else:
            print("Predicción final: CLIENTE NO SE SUSCRIBIRÁ")

        return prob, pred

    except Exception as e:

        print("Error durante la predicción:")
        print(e)

        return None, None