import streamlit as st
import pandas as pd
from utils.sql_data_connector import load_utilization_vs_prediction
from utils.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

st.set_page_config(page_title="Prediction vs Actuals", layout="wide")

st.title("🔍 Actual vs Predicted Performance")
st.markdown("Compare actual utilization data with model predictions to evaluate performance.")

# Sidebar filters
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2025-07-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-08-15"))

# Load data
df = load_utilization_vs_prediction(start_date, end_date)

if df.empty:
    st.warning("No data found for the selected date range.")
    st.stop()

# Metrics
st.subheader("📊 Model Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    mae = mean_absolute_error(df["ActualHours"], df["PredictedHours"])
    st.metric("MAE (Hours)", f"{mae:.2f}")

with col2:
    rmse = root_mean_squared_error(df["ActualHours"], df["PredictedHours"])
    st.metric("RMSE (Hours)", f"{rmse:.2f}")

with col3:
    mape = mean_absolute_percentage_error(df["ActualHours"], df["PredictedHours"])
    st.metric("MAPE (Hours)", f"{mape:.2f}%")

# Visual comparison
st.subheader("📈 Predicted vs Actual (Hours)")

selected_punchcodes = st.multiselect(
    "Select PunchCodes to view",
    options=df["PunchCode"].unique(),
    default=df["PunchCode"].unique()[:10]  # Default to first 10 PunchCodes,
)

filtered_df = df[df["PunchCode"].isin(selected_punchcodes)]

for punchcode in selected_punchcodes:
    st.write(f"### PunchCode: {punchcode}")
    pc_df = filtered_df[filtered_df["PunchCode"] == punchcode]
    chart_df = pc_df[["Date", "ActualHours", "PredictedHours"]].set_index("Date").sort_index()
    st.line_chart(chart_df)

# Raw data
with st.expander("🔍 View Raw Data"):
    st.dataframe(df)
