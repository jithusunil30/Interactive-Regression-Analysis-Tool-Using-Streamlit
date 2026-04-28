import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px

from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    accuracy_score, precision_score,
    recall_score, f1_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RegStatsAI", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 RegStatsAI")
page = st.sidebar.radio("Navigation", ["Home", "Analysis", "Results"])

# ---------------- SESSION ----------------
if "df" not in st.session_state:
    st.session_state.df = None
if "result" not in st.session_state:
    st.session_state.result = None

# ---------------- HOME ----------------
if page == "Home":

    st.title("🚀 Regression & Statistical Modeling Platform")

    st.markdown("""
    Upload your dataset and perform:
    - Regression Analysis
    - Statistical Testing
    - AI Insights
    """)

    uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded).dropna()
        st.session_state.df = df
        st.success("Dataset Uploaded Successfully")

# ---------------- ANALYSIS ----------------
if page == "Analysis" and st.session_state.df is not None:

    df = st.session_state.df
    columns = df.columns.tolist()

    st.header("⚙️ Model Configuration")

    # -------- AUTO MODEL DETECTION --------
    y_var = st.selectbox("Select Target Variable", columns)
    x_vars = st.multiselect("Select Features", columns)

    auto = st.checkbox("🤖 Auto Select Best Model")

    if auto and y_var:
        if df[y_var].nunique() == 2:
            model_type = "LOGISTIC"
        else:
            model_type = "MLR"
        st.info(f"Auto Selected Model: {model_type}")
    else:
        model_type = st.selectbox("Model Type", ["SLR", "MLR", "LOGISTIC"])

    # -------- DATA PREVIEW --------
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # -------- CORRELATION --------
    st.subheader("📊 Correlation Matrix")
    st.dataframe(df.corr())

    # -------- GRAPH --------
    if len(x_vars) > 0:
        st.subheader("📈 Visualization")
        fig = px.scatter(df, x=x_vars[0], y=y_var, trendline="ols")
        st.plotly_chart(fig)

    # -------- RUN MODEL --------
    if st.button("🚀 Run Model"):

        y = df[y_var]
        X = df[x_vars]
        X = sm.add_constant(X)

        result = {}

        # -------- LINEAR --------
        if model_type in ["SLR", "MLR"]:
            model = sm.OLS(y, X).fit()
            y_hat = model.predict(X)

            params = model.params

            equation = f"{y_var} = "
            for name in params.index:
                coef = params[name]
                if name == "const":
                    equation += f"{round(coef,4)} "
                else:
                    equation += f"+ ({round(coef,4)} × {name}) "

            # INSIGHTS
            insights = []
            for var, p in model.pvalues.items():
                if p < 0.05:
                    insights.append(f"✅ {var} is significant")
                else:
                    insights.append(f"❌ {var} is NOT significant")

            result = {
                "type": model_type,
                "equation": equation,
                "r2": model.rsquared,
                "adj_r2": model.rsquared_adj,
                "coef": params.to_dict(),
                "insights": insights
            }

        # -------- LOGISTIC --------
        if model_type == "LOGISTIC":
            model = sm.Logit(y, X).fit(disp=0)

            prob = model.predict(X)
            y_hat = (prob > 0.5).astype(int)

            result = {
                "type": "LOGISTIC",
                "accuracy": accuracy_score(y, y_hat),
                "precision": precision_score(y, y_hat),
                "recall": recall_score(y, y_hat),
                "f1": f1_score(y, y_hat),
                "auc": roc_auc_score(y, prob),
                "confusion": confusion_matrix(y, y_hat)
            }

        st.session_state.result = result
        st.success("Model Trained Successfully!")

# ---------------- RESULTS ----------------
if page == "Results" and st.session_state.result:

    res = st.session_state.result

    st.header("📊 Model Results")

    # -------- LINEAR --------
    if res["type"] != "LOGISTIC":

        st.subheader("📌 Regression Equation")
        st.write(res["equation"])

        col1, col2 = st.columns(2)
        col1.metric("R²", round(res["r2"], 4))
        col2.metric("Adjusted R²", round(res["adj_r2"], 4))

        st.subheader("📊 Coefficients")
        st.json(res["coef"])

        st.subheader("🧠 AI Insights")
        for i in res["insights"]:
            st.write(i)

    # -------- LOGISTIC --------
    else:
        st.subheader("📊 Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(res["accuracy"], 4))
        col2.metric("Precision", round(res["precision"], 4))
        col3.metric("Recall", round(res["recall"], 4))

        st.metric("F1 Score", round(res["f1"], 4))
        st.metric("AUC", round(res["auc"], 4))

        st.subheader("📉 Confusion Matrix")
        st.write(res["confusion"])

    # -------- DOWNLOAD --------
    st.download_button(
        "📥 Download Results",
        data=str(res),
        file_name="results.txt"
    )