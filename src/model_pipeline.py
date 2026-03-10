"""
Pipeline de Preprocesamiento — Customer Shopping Dataset

Descripción:
Módulo encargado de aplicar las transformaciones de datos necesarias antes
del entrenamiento o uso del modelo de Machine Learning. Incluye limpieza,
ingeniería de características y codificación de variables categóricas.

Objetivo:
Preparar el dataset para que pueda ser utilizado por el modelo predictivo,
garantizando que las variables tengan el formato adecuado.

Proceso:
1. Copia del dataset original para evitar modificar los datos fuente.
2. Creación de nuevas variables (feature engineering), como indicadores
   de descuento y métricas de interacción entre variables.
3. Conversión de variables categóricas binarias a formato numérico.
4. Transformación de la frecuencia de compra a una escala numérica.
5. Eliminación de columnas irrelevantes para el modelo.
6. Aplicación de One-Hot Encoding a variables categóricas nominales.

Variables generadas:
- Has_Discount
- Gender_Is_Male
- Frequency_Numeric
- Age_x_Freq
- Total_Engagement

Resultado:
DataFrame preprocesado listo para ser utilizado en el entrenamiento
o predicción del modelo de Machine Learning.
"""

import pandas as pd

FREQ_MAP = {
    'Weekly': 52,
    'Fortnightly': 26,
    'Bi-Weekly': 26,
    'Monthly': 12,
    'Every 3 Months': 4,
    'Quarterly': 4,
    'Annually': 1
}

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline de preprocesamiento utilizado
    en el entrenamiento del modelo.
    """

    print("🔄 Iniciando preprocesamiento...")

    d = df.copy()

    # Feature ingeniería básica
    d['Has_Discount'] = (d['Discount Applied'] == 'Yes').astype(int)

    # Eliminación columnas irrelevantes
    d = d.drop(columns=['Customer ID', 'Item Purchased', 'Discount Applied'], errors='ignore')

    # Variables binarias
    d['Gender_Is_Male'] = (d['Gender'] == 'Male').astype(int)
    d['Subscription Status'] = (d['Subscription Status'] == 'Yes').astype(int)
    d['Promo Code Used'] = (d['Promo Code Used'] == 'Yes').astype(int)

    # Frecuencia numérica
    d['Frequency_Numeric'] = d['Frequency of Purchases'].map(FREQ_MAP)

    # Interacciones
    d['Age_x_Freq'] = d['Age'] * d['Frequency_Numeric']
    d['Total_Engagement'] = d['Previous Purchases'] * d['Review Rating']

    # One-Hot Encoding
    nominal_cols = [
        'Category',
        'Season',
        'Size',
        'Shipping Type',
        'Payment Method'
    ]

    d = pd.get_dummies(d, columns=nominal_cols, drop_first=True)

    print("✅ Preprocesamiento finalizado.")
    return d