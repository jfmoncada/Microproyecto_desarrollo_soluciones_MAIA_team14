"""
Dashboard Analítico — Customer Shopping Behavior

Descripción:
Aplicación interactiva desarrollada en Dash para la exploración visual y análisis
del comportamiento de compra de clientes. El dashboard permite filtrar dinámicamente
los datos y visualizar métricas clave de negocio para facilitar la toma de decisiones.

Objetivos:
- Analizar patrones de compra por categoría, edad y suscripción
- Identificar tendencias de consumo y segmentación de clientes
- Visualizar métricas clave (KPIs) en tiempo real
- Detectar relaciones entre variables relevantes del dataset

Funcionalidades principales:
- Filtros interactivos por categoría y rango de edad
- Indicadores KPI dinámicos
- Visualizaciones estadísticas y comparativas
- Análisis de dispersión, distribución y probabilidades
- Ranking de categorías por gasto promedio

Resultados esperados:
- Comprensión clara del comportamiento del cliente
- Insights accionables para negocio
- Base visual para reportes ejecutivos
- Soporte para futuras fases de modelado predictivo

Tecnologías:
Dash · Plotly · Pandas · Bootstrap · Python
"""

# src/dashboard_jm.py

import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

from predict_utils import predict_subscription

# ======================================================
# CARGAR DATOS
# ======================================================

DATA_PATH = "data/raw/shopping_behavior_updated.csv"
df = pd.read_csv(DATA_PATH)

# ======================================================
# ESTILO VISUAL
# ======================================================

STYLE = {
    'bg': '#0B0F14',
    'card': '#121821',
    'accent': '#22C55E',
    'accent_soft': '#4ADE80',
    'text': '#E6EDF3',
    'muted': '#94A3B8',
    'grid': '#1E293B'
}

external_stylesheets = [dbc.themes.CYBORG]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets
)

server = app.server

# ======================================================
# COMPONENTE KPI
# ======================================================

def kpi_box(label, id):
    return html.Div([
        html.P(label, style={
            'color': STYLE['muted'],
            'fontSize': '10px',
            'marginBottom': '2px'
        }),
        html.H3(id=id, style={
            'color': STYLE['accent'],
            'margin': '0px'
        })
    ], style={
        'padding': '10px'
    })


# ======================================================
# LAYOUT
# ======================================================

app.layout = html.Div(
    style={
        'backgroundColor': STYLE['bg'],
        'minHeight': '100vh',
        'padding': '20px'
    },
    children=[

        html.H3("📊 Sistema Analítico y Predictivo Retail",
                style={'color': STYLE['accent']}),

        html.Hr(),

        dbc.Row([

            # ================= SIDEBAR =================
            dbc.Col([

                html.H5("Filtros Analíticos"),

                dcc.Dropdown(
                    id='category-filter',
                    options=[{'label': c, 'value': c}
                             for c in df['Category'].unique()],
                    value=df['Category'].unique()[0],
                    placeholder="Categoría"
                ),

                dcc.RangeSlider(
                    id='age-slider',
                    min=df['Age'].min(),
                    max=df['Age'].max(),
                    value=[df['Age'].min(), df['Age'].max()],
                    marks={i: str(i) for i in range(20, 71, 10)}
                ),

                html.Hr(),

                # ================= PREDICCIÓN =================

                html.H5("🔮 Predicción Individual"),

                dbc.Input(
                    id="age-input",
                    type="number",
                    placeholder="Edad",
                    className="mb-2"
                ),

                dcc.Dropdown(
                    id="gender-input",
                    options=[
                        {"label": "Masculino", "value": "Male"},
                        {"label": "Femenino", "value": "Female"}
                    ],
                    placeholder="Género",
                    className="mb-2"
                ),

                dcc.Dropdown(
                    id="category-input",
                    options=[{"label": i, "value": i} for i in df["Category"].unique()],
                    placeholder="Categoría de compra",
                    className="mb-2"
                ),

                dcc.Dropdown(
                    id="season-input",
                    options=[{"label": i, "value": i} for i in df["Season"].unique()],
                    placeholder="Temporada",
                    className="mb-2"
                ),

                dcc.Dropdown(
                    id="size-input",
                    options=[{"label": i, "value": i} for i in df["Size"].unique()],
                    placeholder="Talla",
                    className="mb-2"
                ),

                dcc.Dropdown(
                    id="shipping-input",
                    options=[{"label": i, "value": i} for i in df["Shipping Type"].unique()],
                    placeholder="Tipo de envío",
                    className="mb-2"
                ),

                dcc.Dropdown(
                    id="payment-input",
                    options=[{"label": i, "value": i} for i in df["Payment Method"].unique()],
                    placeholder="Método de pago",
                    className="mb-2"
                ),

                dcc.Dropdown(
                    id="frequency-input",
                    options=[{"label": i, "value": i} for i in df["Frequency of Purchases"].unique()],
                    placeholder="Frecuencia de compra",
                    className="mb-2"
                ),

                dbc.Input(
                    id="previous-input",
                    type="number",
                    placeholder="Compras previas",
                    className="mb-2"
                ),

                dbc.Input(
                    id="rating-input",
                    type="number",
                    placeholder="Calificación (1-5)",
                    className="mb-2"
                ),

                dcc.Dropdown(
                    id="promo-input",
                    options=[
                        {"label": "Sí", "value": "Yes"},
                        {"label": "No", "value": "No"}
                    ],
                    placeholder="Usó código promocional",
                    className="mb-2"
                ),

                dcc.Dropdown(
                    id="discount-input",
                    options=[
                        {"label": "Sí", "value": "Yes"},
                        {"label": "No", "value": "No"}
                    ],
                    placeholder="Se aplicó descuento",
                    className="mb-2"
                ),

                dbc.Button(
                    "Predecir Suscripción",
                    id="predict-btn",
                    color="success",
                    className="mb-2"
                ),

                html.Div(id="prediction-output")

            ], md=3),

            # ================= PANEL PRINCIPAL =================

            dbc.Col([

                dbc.Row([
                    dbc.Col(kpi_box("Registros", "kpi-total")),
                    dbc.Col(kpi_box("Ticket Promedio", "kpi-avg")),
                    dbc.Col(kpi_box("% Suscripción", "kpi-sub")),
                ]),

                dcc.Graph(id='subscription-plot'),
                dcc.Graph(id='amount-dist-plot')

            ], md=9)

        ])
    ]
)

# ======================================================
# CALLBACK DASHBOARD ANALÍTICO
# ======================================================

@app.callback(
    [
        Output('subscription-plot', 'figure'),
        Output('amount-dist-plot', 'figure'),
        Output('kpi-total', 'children'),
        Output('kpi-avg', 'children'),
        Output('kpi-sub', 'children')
    ],
    [
        Input('category-filter', 'value'),
        Input('age-slider', 'value')
    ]
)
def update_dashboard(cat, age_range):

    dff = df[
        (df['Age'] >= age_range[0]) &
        (df['Age'] <= age_range[1]) &
        (df['Category'] == cat)
    ]

    # KPIs
    total = len(dff)
    avg = dff['Purchase Amount (USD)'].mean()
    sub_rate = (dff['Subscription Status'] == 'Yes').mean()

    # Gráfico Suscripción
    fig1 = px.histogram(
        dff,
        x="Gender",
        color="Subscription Status",
        barmode="group",
        title="Suscripción por Género"
    )

    # Distribución Compra
    fig2 = px.histogram(
        dff,
        x="Purchase Amount (USD)",
        title="Distribución del Monto de Compra"
    )

    return (
        fig1,
        fig2,
        f"{total}",
        f"${avg:.2f}" if not pd.isna(avg) else "$0",
        f"{sub_rate:.0%}" if not pd.isna(sub_rate) else "0%"
    )

# ======================================================
# CALLBACK PREDICCIÓN
# ======================================================

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("age-input", "value"),
    State("gender-input", "value"),
    State("category-input", "value"),
    State("season-input", "value"),
    State("size-input", "value"),
    State("shipping-input", "value"),
    State("payment-input", "value"),
    State("frequency-input", "value"),
    State("previous-input", "value"),
    State("rating-input", "value"),
    State("promo-input", "value"),
    State("discount-input", "value"),
)
def run_prediction(
    n_clicks,
    age,
    gender,
    category,
    season,
    size,
    shipping,
    payment,
    frequency,
    previous,
    rating,
    promo,
    discount
):
    
    print("BOTÓN PRESIONADO")
    print(age, gender, category, season, size, shipping,
          payment, frequency, previous, rating, promo, discount)

    if not n_clicks:
        return ""

    valores = [
        age, gender, category, season, size,
        shipping, payment, frequency,
        previous, rating, promo, discount
    ]

    if any(v is None for v in valores):
        return html.P("⚠️ Complete todas las variables para predecir.")

    input_dict = {
        "Age": age,
        "Gender": gender,
        "Category": category,
        "Season": season,
        "Size": size,
        "Shipping Type": shipping,
        "Payment Method": payment,
        "Frequency of Purchases": frequency,
        "Previous Purchases": previous,
        "Review Rating": rating,
        "Promo Code Used": promo,
        "Discount Applied": discount
    }

    prob, pred = predict_subscription(input_dict)

    if prob > 0.75:
        estrategia = "Alta probabilidad — Priorizar campaña premium"
        color = "lime"
    elif prob > 0.5:
        estrategia = "Probabilidad media — Ofrecer descuento"
        color = "orange"
    else:
        estrategia = "Baja probabilidad — Estrategia de awareness"
        color = "red"

    return html.Div([
        html.H4(
            f"Resultado: {'Suscrito' if pred == 1 else 'No Suscrito'}"
        ),
        html.P(f"Probabilidad estimada: {prob:.2%}"),
        html.P(
            f"Estrategia sugerida: {estrategia}",
            style={"color": color}
        )
    ])

# ======================================================
# RUN APP
# ======================================================

if __name__ == "__main__":
    #app.run(debug=True)
    #app.run(host="0.0.0.0", port=8050, debug=True)
    app.run(debug=False, host="0.0.0.0", port=8050)