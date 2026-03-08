import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(page_title="Smart Home Energy Predictor", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #F4F6F9;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background-color:#2E86C1;padding:15px;border-radius:10px">
        <h1 style="color:white;text-align:center;">
         Smart Home Energy Consumption Predictor
        </h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("Upload your Smart Home dataset (CSV or Excel).")

# =============================
# FILE UPLOAD
# =============================

uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
REQUIRED_COLUMNS = ['date', 'Appliances', 'T_out', 'RH_out', 'Windspeed']

# =============================
# CLEANING
# =============================

def clean_data(df):
    df = df.drop_duplicates()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')
    if 'rv2' in df.columns:
        df = df.drop(columns=['rv2'])
    return df

# =============================
# FEATURE ENGINEERING
# =============================

def feature_engineering(df):
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['weekend'] = df['date'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)

    df['Appliances_lag_1'] = df['Appliances'].shift(1)
    df['Appliances_rolling_mean_6'] = df['Appliances'].rolling(window=6).mean()

    df = df.dropna()
    return df

# =============================
# LOAD MODEL
# =============================

@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

# =============================
# MAIN LOGIC
# =============================

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)


        if not all(col in df.columns for col in REQUIRED_COLUMNS):
            st.error("Uploaded file missing required columns.")
            st.stop()

        # =============================
        # RAW DATA PREVIEW
        # =============================

        st.markdown("---")
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        df = clean_data(df)
        df = feature_engineering(df)

        # =============================
        # PROCESSED DATA PREVIEW
        # =============================

        st.markdown("---")
        st.subheader(" Processed Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        model = load_model()
        X = df[model.feature_names_in_]
        df["Predicted_Energy"] = model.predict(X)

        # =============================
        # CALCULATIONS
        # =============================

        hourly_usage = df.groupby("hour")["Predicted_Energy"].mean()
        peak_hour = hourly_usage.idxmax()

        def format_hour(h):
            if h == 0:
                return "12 AM - 1 AM"
            elif h < 12:
                return f"{h} AM - {h+1} AM"
            elif h == 12:
                return "12 PM - 1 PM"
            else:
                return f"{h-12} PM - {h-11} PM"

        formatted_peak = format_hour(peak_hour)

        df["day_name"] = df["date"].dt.day_name()
        high_day = df.groupby("day_name")["Predicted_Energy"].mean().idxmax()

        # =============================
        # KEY METRIC CARDS
        # =============================

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg,#E3F2FD,#FFFFFF);
                    padding:25px;border-radius:15px;
                    box-shadow:0px 8px 18px rgba(0,0,0,0.08);
                    text-align:center;">
                    <h4 style="color:#1565C0;">Average Actual Consumption</h4>
                    <h2 style="color:#0D47A1;">{round(df['Appliances'].mean(),2)} Wh</h2>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg,#E8F5E9,#FFFFFF);
                    padding:25px;border-radius:15px;
                    box-shadow:0px 8px 18px rgba(0,0,0,0.08);
                    text-align:center;">
                    <h4 style="color:#2E7D32;">Average Predicted Consumption</h4>
                    <h2 style="color:#1B5E20;">{round(df['Predicted_Energy'].mean(),2)} Wh</h2>
                </div>
            """, unsafe_allow_html=True)

        # =============================
        # ENERGY TREND (BLUE)
        # =============================

        st.markdown("---")
        st.subheader("Energy Trend Over Time")

        trend_df = df.reset_index()
        fig_trend = px.line(
            trend_df,
            x="date",
            y="Predicted_Energy",
            color_discrete_sequence=["#1E88E5"]
        )
        fig_trend.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # =============================
        # HOURLY ENERGY USAGE (GRADIENT)
        # =============================
        st.markdown("---")
        st.subheader("Hourly Energy Usage")

        hourly_df = hourly_usage.reset_index()
        hourly_df.columns = ["Hour", "Predicted_Energy"]

        fig_bar = px.bar(
            hourly_df,
            x="Hour",
            y="Predicted_Energy",
            color="Predicted_Energy",
            color_continuous_scale=[
                [0.0, "#BBDEFB"],   # Light Blue
                [1.0, "#0D47A1"]    # Dark Blue
            ]
        )

        fig_bar.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis_title="Hour of Day",
            yaxis_title="Predicted Energy (Wh)"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        # =============================
        # PEAK & HIGH USAGE CARDS
        # =============================

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg,#FFF3E0,#FFFFFF);
                    padding:20px;border-radius:15px;
                    box-shadow:0px 6px 15px rgba(0,0,0,0.08);
                    text-align:center;">
                    <h4 style="color:#EF6C00;">Peak Hour</h4>
                    <h3 style="color:#E65100;">{formatted_peak}</h3>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg,#F3E5F5,#FFFFFF);
                    padding:20px;border-radius:15px;
                    box-shadow:0px 6px 15px rgba(0,0,0,0.08);
                    text-align:center;">
                    <h4 style="color:#6A1B9A;">High Usage Day</h4>
                    <h3 style="color:#4A148C;">{high_day}</h3>
                </div>
            """, unsafe_allow_html=True)

        # =============================
        # DOWNLOAD
        # =============================

        st.markdown("---")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="predicted_energy.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a dataset to begin prediction.")