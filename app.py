# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="A Visual Record of Loss",
    page_icon="🇵🇸",
    layout="wide"
)

# --- Data and Model Loading ---
@st.cache_data
def load_historical_data():
    path = Path("data/processed/monthly_fatalities_timeseries.csv")
    if not path.exists():
        st.error("Processed data file not found. Please run 'dvc repro' first.")
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    return df

@st.cache_resource
def load_model():
    path = Path("models/forecasting_model.pkl")
    if not path.exists():
        st.error("Model file not found. Please run 'dvc repro' first.")
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

historical_df = load_historical_data()
model = load_model()

# --- Main Application ---
st.title("A Visual Record of Loss")
st.markdown("### A Time-Series Forecast of Palestinian Fatalities (2000-2023)")

if historical_df is not None and model is not None:
    # --- Key Statistics Section ---
    # This section provides immediate, powerful numbers for context.
    total_fatalities = historical_df['fatalities'].sum()
    average_monthly_fatalities = historical_df['fatalities'].mean()
    highest_month_value = historical_df['fatalities'].max()
    highest_month_date = historical_df.loc[historical_df['fatalities'].idxmax()]['date'].strftime('%B %Y')

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Palestinian Fatalities Recorded", f"{total_fatalities:,}")
    col2.metric("Average Fatalities Per Month", f"{average_monthly_fatalities:.1f}")
    col3.metric(f"Highest Single Month ({highest_month_date})", f"{highest_month_value:,} fatalities")
    st.markdown("---")

    # --- Forecasting ---
    forecast_years = st.slider("Select years to forecast into the future:", 1, 20, 5, key="forecast_slider")
    forecast_steps = forecast_years * 12

    forecast = model.get_forecast(steps=forecast_steps)
    forecast_df = forecast.summary_frame(alpha=0.05) # 95% confidence interval

    future_dates = pd.date_range(start=historical_df['date'].iloc[-1], periods=forecast_steps + 1, freq='M')[1:]
    forecast_df.index = future_dates

    # --- Visualization ---
    st.header("Historical Data vs. Forecasted Trend")
    st.markdown("""
    The chart below shows the actual number of Palestinian fatalities per month in **<span style='color:green;'>green</span>**. 
    The dotted **<span style='color:red;'>red</span>** line is the model's projection of what the future could look like if historical trends of conflict continue. 
    **Crucially, annotations have been added to mark major military operations, providing context for the largest spikes in loss of life.**
    """, unsafe_allow_html=True)

    fig = go.Figure()

    # Plot historical data in GREEN
    fig.add_trace(go.Scatter(
        x=historical_df['date'],
        y=historical_df['fatalities'],
        mode='lines',
        name='Historical Fatalities',
        line=dict(color='red', width=2.5)
    ))

    # Plot forecasted data in RED
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['mean'],
        mode='lines',
        name='Forecasted Trend',
        line=dict(color='white', dash='dot', width=2.5)
    ))

    # Plot confidence interval with a semi-transparent RED fill
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['mean_ci_upper'],
        mode='lines', name='Upper Confidence Interval', line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df['mean_ci_lower'],
        mode='lines', name='Lower Confidence Interval', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', showlegend=False
    ))

    # --- NEW: Add Annotations for Context ---
    # These annotations turn abstract spikes into historical events.
    annotations = [
        {'date': '2009-01-01', 'text': 'Gaza War (Op. Cast Lead)'},
        {'date': '2012-11-01', 'text': 'Op. Pillar of Defense'},
        {'date': '2014-07-01', 'text': 'Gaza War (Op. Protective Edge)'},
    ]

    for event in annotations:
        fig.add_annotation(
            x=event['date'], y=historical_df[historical_df['date'].dt.strftime('%Y-%m') == event['date'][:7]]['fatalities'].max(),
            text=event['text'], showarrow=True, arrowhead=2, ax=0, ay=-60,
            font=dict(color="white", size=12), bgcolor="black", borderpad=4
        )
    
    # --- Layout and Style Updates ---
    fig.update_layout(
        title_text="Monthly Palestinian Fatalities: Historical vs. Forecast",
        xaxis_title="Year",
        yaxis_title="Number of Fatalities",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Disclaimer & Ethical Considerations:** This forecast is a mathematical projection based on historical data. It **cannot** predict specific future events like peace treaties or major escalations. Its purpose is to illustrate the potential long-term human cost of the conflict if the underlying conditions remain unchanged.
    """)
else:
    st.warning("Data or model files are missing. Please ensure 'dvc repro' has been run successfully.")