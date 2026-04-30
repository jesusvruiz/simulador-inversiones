import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import base64

st.set_page_config(page_title="Simulador de Inversiones", layout="wide")

# Encabezado profesional
st.markdown("""
<div style="text-align: center; margin-top: -25px;">

<h1 style="
    font-family: 'Segoe UI', sans-serif;
    font-size: 52px;
    font-weight: 900;
    letter-spacing: 2px;
    margin-bottom: 5px;
    background: linear-gradient(90deg, #ffffff, #4da3ff, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;">
    SIMULADOR DE INVERSIONES
</h1>

<div style="margin-top: 20px; display: flex; justify-content: center; align-items: center; gap: 10px;">

<span style="
    font-family: 'Segoe UI', sans-serif;
    font-size: 34px;
    font-weight: 800;">
    Jesús Victoriano Ruiz
</span>

<a href="https://www.linkedin.com/in/jesus-angel-victoriano-ruiz-133294182/" target="_blank"
style="text-decoration: none;">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="26">
</a>

</div>

<p style="
    font-family: 'Segoe UI', sans-serif;
    font-size: 18px;
    font-weight: 600;
    margin-top: 6px;">
    Estudiante de Economía - Universidad Nacional de Piura
</p>

<p style="
    font-family: 'Segoe UI', sans-serif;
    font-size: 17px;
    max-width: 900px;
    margin: 15px auto 0 auto;
    line-height: 1.7;">
    Aplicación web orientada al análisis financiero y de datos, desarrollada para simular portafolios de inversión,
    evaluar rendimiento y riesgo, y comparar resultados frente a benchmarks mediante métricas como volatilidad,
    Sharpe Ratio y beta.
</p>

</div>
""", unsafe_allow_html=True)
# Sidebar
st.sidebar.header("Configuración del portafolio")

tickers_input = st.sidebar.text_input(
    "Ingresa los tickers separados por comas",
    value="AAPL,MSFT,TSLA"
)

start_date = st.sidebar.date_input(
    "Fecha de inicio",
    value=pd.to_datetime("2023-01-01")
)

end_date = st.sidebar.date_input(
    "Fecha de fin",
    value=pd.to_datetime("today")
)

initial_investment = st.sidebar.number_input(
    "Monto inicial a invertir",
    min_value=100,
    value=1000,
    step=100
)

benchmark = st.sidebar.selectbox(
    "Selecciona benchmark",
    ["SPY (EE.UU.)", "EPU (Perú)"]
)

if benchmark == "SPY (EE.UU.)":
    benchmark_symbol = "SPY"
else:
    benchmark_symbol = "EPU"

# Procesar tickers
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

if len(tickers) == 0:
    st.warning("Por favor, ingresa al menos un ticker.")
    st.stop()

# Descargar datos
try:
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)["Close"]
except Exception as e:
    st.error(f"Error al descargar datos: {e}")
    st.stop()

# Descargar datos del benchmark
try:
    market_data = yf.download(
        benchmark_symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )["Close"]
except Exception as e:
    st.error(f"Error al descargar datos del benchmark: {e}")
    st.stop()

market_returns = market_data.pct_change().dropna()

# Si solo hay una acción, convertir a DataFrame
if isinstance(data, pd.Series):
    data = data.to_frame()

# Eliminar columnas totalmente vacías
data = data.dropna(axis=1, how="all")

if data.empty:
    st.error("No se pudieron descargar datos. Verifica que los tickers ingresados sean correctos.")
    st.stop()

# Avisar si algún ticker no se descargó
valid_tickers = list(data.columns)
missing_tickers = [t for t in tickers if t not in valid_tickers]

if missing_tickers:
    st.warning(f"No se pudieron descargar estos tickers: {', '.join(missing_tickers)}")

st.subheader("Precios históricos")
st.dataframe(data.tail())

# Pesos personalizados
st.sidebar.subheader("Pesos del portafolio (%)")

weights_list = []
default_weight = round(100 / len(valid_tickers), 2)

for ticker in valid_tickers:
    weight = st.sidebar.number_input(
        f"Peso de {ticker}",
        min_value=0.0,
        max_value=100.0,
        value=default_weight,
        step=1.0
    )
    weights_list.append(weight)

weights = np.array(weights_list)

if abs(weights.sum() - 100) > 0.01:
    st.error(f"La suma de los pesos debe ser 100%. Actualmente es {weights.sum():.2f}%.")
    st.stop()

weights = weights / 100

# Calcular rendimientos diarios
returns = data.pct_change().dropna()

if returns.empty:
    st.error("No hay suficientes datos para calcular rendimientos.")
    st.stop()

# Rendimiento del portafolio
portfolio_returns = returns.dot(weights)

if portfolio_returns.empty:
    st.error("No se pudo calcular el rendimiento del portafolio.")
    st.stop()

# Alinear fechas entre portafolio y mercado
combined = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
combined.columns = ["Portfolio", "Market"]

# Calcular beta
covariance = combined["Portfolio"].cov(combined["Market"])
market_variance = combined["Market"].var()

beta = covariance / market_variance if market_variance != 0 else 0

# Volatilidad anual
volatility = portfolio_returns.std() * np.sqrt(252) * 100
# Sharpe Ratio
risk_free_rate = 0.04
annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)

sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
# Evolución del valor del portafolio
portfolio_growth = (1 + portfolio_returns).cumprod() * initial_investment

if portfolio_growth.empty:
    st.error("No se pudo calcular el crecimiento del portafolio.")
    st.stop()

# Mostrar gráfico
# Simular crecimiento del benchmark con el mismo monto inicial
market_growth = (1 + market_returns).cumprod() * initial_investment

# Unir portafolio y benchmark
comparison_df = pd.concat([portfolio_growth, market_growth], axis=1).dropna()
comparison_df.columns = ["Portafolio", benchmark_symbol]

# Reiniciar índice y renombrar columna de fecha
comparison_df = comparison_df.reset_index()
comparison_df = comparison_df.rename(columns={comparison_df.columns[0]: "Fecha"})

# Convertir a formato largo
comparison_plot = comparison_df.melt(
    id_vars="Fecha",
    value_vars=["Portafolio", benchmark_symbol],
    var_name="Serie",
    value_name="Valor"
)

# Gráfico comparativo
fig = px.line(
    comparison_plot,
    x="Fecha",
    y="Valor",
    color="Serie",
    title=f"Comparación: Portafolio vs Benchmark ({benchmark_symbol})"
)

fig.update_layout(
    template="plotly_white",
    legend_title="",
    xaxis_title="Fecha",
    yaxis_title="Valor del portafolio",
    title_x=0.5
)

st.plotly_chart(fig, width="stretch")
st.markdown("---")

st.subheader("Resumen por activo")

asset_returns = (data.iloc[-1] / data.iloc[0] - 1) * 100
asset_volatility = returns.std() * np.sqrt(252) * 100

summary_df = pd.DataFrame({
    "Activo": asset_returns.index,
    "Peso (%)": weights * 100,
    "Retorno (%)": asset_returns.values,
    "Volatilidad anual (%)": asset_volatility.values
})

summary_df = summary_df.round(2)

st.dataframe(summary_df)

csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar resumen en CSV",
    data=csv,
    file_name="resumen_portafolio.csv",
    mime="text/csv"
)

st.markdown("---")

st.subheader("Correlación entre activos")

corr = returns.corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu",
    title="Mapa de correlación"
)

fig_corr.update_layout(
    template="plotly_white",
    title_x=0.5
)

st.plotly_chart(fig_corr, width="stretch")

benchmark_final = comparison_df[benchmark_symbol].iloc[-1]
benchmark_return = (benchmark_final / initial_investment - 1) * 100
# Métricas básicas
total_return = (portfolio_growth.iloc[-1] / initial_investment - 1) * 100

st.markdown("---")

st.subheader("Métricas básicas")
col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Monto final", f"${portfolio_growth.iloc[-1]:,.2f}")
col2.metric("Retorno total", f"{total_return:.2f}%")
col3.metric("Volatilidad anual", f"{volatility:.2f}%")
col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col5.metric("Beta", f"{beta:.2f}")
col6.metric("Retorno Benchmark", f"{benchmark_return:.2f}%")

st.markdown("---")

st.subheader("Interpretación del portafolio")

interpretation = []

if volatility > 25:
    interpretation.append("El portafolio presenta una volatilidad alta, por lo que su nivel de riesgo es elevado.")
else:
    interpretation.append("El portafolio presenta una volatilidad moderada o baja, lo que sugiere un comportamiento relativamente más estable.")

if sharpe_ratio > 1:
    interpretation.append("El Sharpe Ratio indica una relación riesgo-retorno favorable.")
else:
    interpretation.append("El Sharpe Ratio sugiere que el rendimiento obtenido podría no compensar adecuadamente el riesgo asumido.")

if beta > 1:
    interpretation.append(f"La beta es mayor que 1, lo que significa que el portafolio es más sensible que el benchmark ({benchmark_symbol}).")
elif beta < 1:
    interpretation.append(f"La beta es menor que 1, lo que indica que el portafolio es menos volátil que el benchmark ({benchmark_symbol}).")
else:
    interpretation.append(f"La beta cercana a 1 sugiere un comportamiento similar al benchmark ({benchmark_symbol}).")

for text in interpretation:
    st.write("- " + text)

st.markdown("---")

st.subheader("Comparación con el benchmark")

portfolio_final = comparison_df["Portafolio"].iloc[-1]
benchmark_final = comparison_df[benchmark_symbol].iloc[-1]

if portfolio_final > benchmark_final:
    st.success(f"El portafolio superó al benchmark ({benchmark_symbol}) en el periodo analizado.")
elif portfolio_final < benchmark_final:
    st.warning(f"El benchmark ({benchmark_symbol}) tuvo mejor desempeño que el portafolio.")
else:
    st.info("El portafolio tuvo un desempeño similar al benchmark.")

st.markdown("---")

st.subheader("Evaluación de riesgo-retorno")

if sharpe_ratio > 1:
    st.success("El portafolio muestra una buena relación riesgo-retorno.")
else:
    st.warning("El portafolio podría no compensar adecuadamente el riesgo.")
