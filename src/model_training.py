"""
Modelo de Machine Learning — Predicción de Suscripción de Clientes

Descripción:
Script para entrenar un modelo de clasificación que predice si un cliente
tendrá una suscripción a partir de su comportamiento de compra. El proceso
incluye carga de datos, preprocesamiento automático, entrenamiento del modelo,
evaluación y almacenamiento del modelo entrenado.

Objetivos:
- Predecir el estado de suscripción de clientes (Yes / No)
- Automatizar el procesamiento de variables numéricas y categóricas
- Evaluar el modelo mediante métricas de clasificación
- Guardar el modelo entrenado para su uso posterior en APIs o aplicaciones

Proceso:
1. Carga del dataset de comportamiento de compra.
2. Selección de variables predictoras y variable objetivo.
3. Preprocesamiento:
   - Imputación de valores faltantes
   - Codificación One-Hot para variables categóricas
4. División del dataset en entrenamiento (80%) y prueba (20%).
5. Entrenamiento del modelo Random Forest.
6. Evaluación mediante classification_report.
7. Guardado del modelo entrenado en formato .pkl.

Resultados esperados:
- Modelo capaz de predecir la suscripción de clientes.
- Pipeline reutilizable para predicción en producción.

"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ==========================================
# CARGAR DATA
# ==========================================

print("📥 Cargando datos...")

df = pd.read_csv("data/raw/shopping_behavior_updated.csv")

print("✅ Datos cargados correctamente")
print(f"Registros: {len(df)}")

# ==========================================
# DEFINIR VARIABLES
# ==========================================

#X = df.drop("Subscription Status", axis=1)
X = df[
[
"Age",
"Gender",
"Category",
"Season",
"Size",
"Shipping Type",
"Payment Method",
"Frequency of Purchases",
"Previous Purchases",
"Review Rating",
"Promo Code Used",
"Discount Applied"
]
]
y = df["Subscription Status"].map({"Yes": 1, "No": 0})

# Separar columnas
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

print("Columnas categóricas:", categorical_cols)
print("Columnas numéricas:", numeric_cols)

# ==========================================
# PREPROCESAMIENTO
# ==========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ==========================================
# PIPELINE COMPLETO
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

print("✂️ Dividiendo datos...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# ENTRENAMIENTO
# ==========================================

print("🚀 Entrenando modelo...")

pipeline.fit(X_train, y_train)

print("✅ Entrenamiento completado")

# ==========================================
# EVALUACIÓN
# ==========================================

y_pred = pipeline.predict(X_test)

print("\n📊 REPORTE DE CLASIFICACIÓN\n")
print(classification_report(y_test, y_pred))

# ==========================================
# GUARDAR MODELO
# ==========================================

joblib.dump(pipeline, "models/subscription_model.pkl")

print("\n💾 Modelo guardado en models/subscription_model.pkl")