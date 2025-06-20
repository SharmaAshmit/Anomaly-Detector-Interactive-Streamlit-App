import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from io import BytesIO


st.set_page_config(page_title="Anomaly Detector", layout="wide")
st.title("📊 Data Anomaly Detection")


uploaded_file = st.file_uploader("📂 Upload your Excel (.xlsx) file", type=["xlsx"])

if uploaded_file is not None:
    
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(df.head())

    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = st.multiselect("🔢 Select numeric columns for anomaly detection:", numeric_cols)

    if selected_features:
        
        st.subheader("🎚️ Set Detection Sensitivity/ Set Threshold")
        contamination_level = st.slider("Select anomaly sensitivity (% of data considered anomalies)", 
                                        min_value=0.01, max_value=0.3, value=0.05, step=0.01)

        
        X = df[selected_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        
        model = IsolationForest(contamination=contamination_level, random_state=42)
        predictions = model.fit_predict(X_scaled)
        df.loc[X.index, 'anomaly'] = pd.Series(predictions, index=X.index).map({1: 'Normal', -1: 'Anomaly'})

        
        total = len(df)
        anomalies = (df['anomaly'] == 'Anomaly').sum()
        st.success(f"✅ Total Records: {total}")
        st.warning(f"🚨 Anomalies Detected: {anomalies}")

        
        anomaly_df = df[df['anomaly'] == 'Anomaly']
        normal_df = df[df['anomaly'] == 'Normal']

        
        tab1, tab2 = st.tabs(["📋 Full Data with Labels", "🚨 Anomalies Only"])
        with tab1:
            st.dataframe(df)
        with tab2:
            st.dataframe(anomaly_df)

        # 
        st.subheader("📈 Anomaly Visualization (Line Chart)")

        if len(selected_features) >= 2:
            x_axis = st.selectbox("📌 Select X-axis Feature", options=selected_features, index=0)
            y_axis = st.selectbox("📌 Select Y-axis Feature", options=[col for col in selected_features if col != x_axis], index=0)
            show_only_anomalies = st.checkbox("🔍 Show Only Anomalies", value=False)

            chart_df = anomaly_df if show_only_anomalies else df

            fig, ax = plt.subplots(figsize=(10, 6))
            for label, color in zip(['Normal', 'Anomaly'], ['blue', 'red']):
                subset = chart_df[chart_df['anomaly'] == label]
                ax.plot(subset[x_axis], subset[y_axis], label=label, color=color, marker='o', alpha=0.7)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("Please select at least two numeric features for visualization.")

        
        @st.cache_data
        def to_excel(dataframe):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                dataframe.to_excel(writer, index=False)
            return output.getvalue()

        st.download_button(
            label="📥 Download Anomalies as Excel",
            data=to_excel(anomaly_df),
            file_name="detected_anomalies.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Please select at least one numeric column to continue.")
else:
    st.info("Upload an Excel (.xlsx) file to begin.")
