import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from io import BytesIO
from streamlit_lottie import st_lottie
import requests
import plotly.express as px
import plotly.io as pio
from datetime import datetime
import base64

# -------------------- Page Config -------------------- #
st.set_page_config(page_title="🚨 Smart Anomaly Detector", layout="wide", page_icon="📊")

# -------------------- Load 3D Animation -------------------- #
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_json = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")

# -------------------- App Header -------------------- #
col1, col2 = st.columns([1, 2])
with col1:
    if lottie_json:
        st_lottie(lottie_json, height=200, key="lottie-anomaly")
with col2:
    st.markdown("""
        <h1 style='color:#4A90E2'>🚨 Smart Anomaly Detection App</h1>
        <p>Upload your dataset and visualize anomalies using Isolation Forest and interactive line charts.</p>
    """, unsafe_allow_html=True)

# -------------------- File Upload -------------------- #
st.markdown("### 📂 Upload Your Excel File")
uploaded_file = st.file_uploader("Upload a .xlsx file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.markdown("### 🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -------------------- Sample Dataset Download -------------------- #
    def get_table_download_link(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="sample_data.csv">📅 Download this preview as CSV</a>'

    st.markdown(get_table_download_link(df), unsafe_allow_html=True)

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = st.multiselect("🎯 Select numeric columns for anomaly detection:", numeric_cols)

    if not selected_features:
        st.warning("⚠️ Please select at least one numeric column.")
        st.stop()

    st.markdown("### 🧪 Anomaly Detection Sensitivity")
    contamination = st.slider("Set contamination (% outliers)", 0.01, 0.3, 0.05, 0.01)

    # Preprocessing
    data = df[selected_features]
    imputer = SimpleImputer(strategy="mean")
    X_clean = imputer.fit_transform(data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Model
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(X_scaled)
    df["Anomaly"] = pd.Series(predictions).map({1: "Normal", -1: "Anomaly"})

    st.success(f"✅ Total Rows: {len(df)} | 🚨 Anomalies Found: {(df['Anomaly'] == 'Anomaly').sum()}")

    # -------------------- Labeled Data Download -------------------- #
    st.download_button("📥 Download Labeled Data", df.to_csv(index=False), file_name="labeled_data.csv")

    tab1, tab2 = st.tabs(["📋 Full Dataset with Labels", "🚨 Only Anomalies"])
    with tab1:
        st.dataframe(df)
    with tab2:
        st.dataframe(df[df["Anomaly"] == "Anomaly"])

    # -------------------- Theme Option -------------------- #
    st.markdown("### 📁 Choose Plot Theme")
    theme = st.radio("Select Theme", ["Dark", "Light"], index=0)
    template = "plotly_dark" if theme == "Dark" else "plotly_white"

    # -------------------- Line Chart -------------------- #
    st.markdown("### 📈 Anomaly Line Chart")

    if len(selected_features) >= 2:
        x_axis = st.selectbox("📌 X-axis Feature", options=selected_features)
        y_axis = st.selectbox("📌 Y-axis Feature", options=[col for col in selected_features if col != x_axis])

        show_only_anomalies = st.checkbox("🔍 Show Only Anomalies", value=False)
        plot_data = df[df["Anomaly"] == "Anomaly"] if show_only_anomalies else df

        fig = px.line(
            plot_data.sort_values(by=x_axis),
            x=x_axis,
            y=y_axis,
            color="Anomaly",
            line_shape="linear",
            color_discrete_map={
                "Normal": "#1f77b4",
                "Anomaly": "#d62728"
            },
            title="📁 Anomaly Detection Line Chart",
            labels={x_axis: f"{x_axis} (X-axis)", y_axis: f"{y_axis} (Y-axis)"},
            template=template
        )

        fig.update_layout(
            font=dict(size=14),
            legend=dict(
                bgcolor="#f9f9f9" if theme == "Light" else "#1e1e1e",
                bordercolor="gray",
                borderwidth=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------- HTML Report -------------------- #
        st.markdown("### 📜 Download HTML Report")

        def generate_html_report(fig, df_summary):
            html_plot = pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset='UTF-8'>
                <title>Anomaly Detection Report</title>
                <style>
                    body {{ background-color: #1e1e1e; color: white; font-family: Arial, sans-serif; padding: 40px; }}
                    h1, h2 {{ color: #4A90E2; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background-color: #2e2e2e; }}
                    th, td {{ border: 1px solid #444; padding: 10px; text-align: left; }}
                </style>
            </head>
            <body>
                <h1>📊 Anomaly Detection Report</h1>
                <h2>🗓️ Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
                <h2>📈 Line Chart</h2>
                {html_plot}
                <h2>📋 Summary</h2>
                <table>
                    <tr><th>Total Rows</th><td>{len(df_summary)}</td></tr>
                    <tr><th>Normal Points</th><td>{(df_summary['Anomaly'] == 'Normal').sum()}</td></tr>
                    <tr><th>Anomalies</th><td>{(df_summary['Anomaly'] == 'Anomaly').sum()}</td></tr>
                </table>
            </body>
            </html>
            """
            return html_template.encode("utf-8")

        html_bytes = generate_html_report(fig, df)
        st.download_button(
            label="📄 Download HTML Report",
            data=html_bytes,
            file_name="anomaly_detection_report.html",
            mime="text/html"
        )
    else:
        st.info("ℹ️ Please select at least two numeric columns for chart generation.")
else:
    st.info("📂 Upload an Excel file to begin.")
