import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
from groq import Groq

from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    accuracy_score, precision_score,
    recall_score, f1_score
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RegStatsAI", layout="wide")

# ---------------- CSS (EXACT UI) ----------------
st.markdown("""
<style>
/* ===== GLOBAL BACKGROUND ===== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #00b4db, #5f8fa6);
    overflow-x: hidden;
}

/* ===== FLOATING NAVBAR ===== */
.navbar {
    position: sticky;
    top: 10px;
    margin: 10px 40px;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(14px);
    border-radius: 14px;
    padding: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
    z-index: 999;

    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}

/* ===== HERO ===== */
.hero {
    text-align: center;
    padding: 100px 20px;
    animation: fadeIn 1.2s ease-in-out;
}

.hero h1 {
    color: white;
    font-size: 48px;
    text-shadow: 0 0 20px rgba(255,255,255,0.6);
}

.hero p {
    color: #e0f2fe;
    font-size: 18px;
}

/* ===== SECTION ===== */
.section {
    text-align: center;
    padding: 70px 20px;
}

/* ===== FEATURES GRID ===== */
.features {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 30px;
    padding: 0 80px;
}

/* ===== GLASS CARDS ===== */
.features .card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 30px;
    color: white;
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);

    transition: 0.4s ease;
    animation: fadeUp 1s ease forwards;
}

/* ===== GLOW EFFECT ===== */
.features .card:hover {
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 0 30px rgba(255,255,255,0.4),
                0 0 60px rgba(0,180,255,0.3);
}

/* ===== TEXT ===== */
.features .card h3 {
    color: white;
    margin-bottom: 10px;
}

.features .card p {
    color: #e0f2fe;
}

/* ===== UPLOAD GLASS ===== */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.15) !important;
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.3);
    transition: 0.3s;
}

/* Hover glow */
[data-testid="stFileUploader"]:hover {
    box-shadow: 0 0 25px rgba(0,180,255,0.5);
}

/* Remove dark layer */
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* Text */
[data-testid="stFileUploader"] * {
    color: white !important;
}

/* ===== SIDEBAR GLASS ===== */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.1) !important;
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255,255,255,0.2);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(40px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ===== REMOVE HEADER ===== */
[data-testid="stHeader"] {
    background: transparent !important;
}
/* ===== CARD SECTIONS ===== */
.card-section {
    border-radius: 16px;
    padding: 25px;
    margin: 25px 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}

/* COLORS */
.blue-card {
    background: linear-gradient(135deg, #6f95e8, #4f7bd9);
}

.green-card {
    background: linear-gradient(135deg, #2f8f1f, #1f6f12);
}

.orange-card {
    background: linear-gradient(135deg, #f2ad66, #e6953c);
}

.white-card {
    background: rgba(255,255,255,0.95);
    color: #111;
}
</style>
""", unsafe_allow_html=True)

# ---------------- AI ----------------
from openai import OpenAI

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="YOUR_API_KEY"   # 🔐 secure
)

def chatbot_response(question, result):

    model_type = result.get("type", "linear")

    if model_type == "logistic":
        performance = f"""
Accuracy: {result.get("accuracy")}
Precision: {result.get("precision")}
Recall: {result.get("recall")}
F1 Score: {result.get("f1")}
"""
    else:
        performance = f"""
R2 Score: {result.get("r2")}
Adjusted R2: {result.get("adj_r2")}
"""

    prompt = f"""
You are a professional data science assistant.

Model Type: {model_type}

Model Performance:
{performance}

Coefficients:
{result.get("coef")}

User Question:
{question}

Give:
- Simple explanation
- Key insights
- Suggestions to improve model
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

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

    st.markdown("""
    <div class="navbar">RegStatsAI</div>

    <div class="hero">
    <h1>Regression And Statistical Modeling Platform</h1>
    <p>Upload your dataset and perform advanced regression analysis, significance testing, and predictive modeling.</p>
    </div>

    <div class="section">
    <h2>Platform Features</h2>

    <div class="features">

    <div class="card">
    <h3>Simple Linear Regression</h3>
    <p>Perform regression analysis with t-test significance testing.</p>
    </div>

    <div class="card">
    <h3>Multiple Linear Regression</h3>
    <p>Analyze relationships between multiple predictors and response variables.</p>
    </div>

    <div class="card">
    <h3>Logistic Regression</h3>
    <p>Predict binary outcomes and analyze classification performance.</p>
    </div>

    <div class="card">
    <h3>Model Diagnostics</h3>
    <p>Residual analysis and normal Q-Q plots to validate regression assumptions.</p>
    </div>

    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Upload Dataset")

    # Custom UI (matches your image)
    st.markdown("""
    <div class="upload-container">
    </div>
    """, unsafe_allow_html=True)

    # REAL uploader (hidden but functional)
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded).dropna()
        st.session_state.df = df
        st.success("Dataset Uploaded Successfully")

# ---------------- ANALYSIS ----------------
if page == "Analysis" and st.session_state.df is not None:

    df = st.session_state.df
    columns = df.columns.tolist()

    st.subheader("⚙️ Model Configuration")

    model_type = st.selectbox(
        "Select Regression Type",
        ["Simple Linear Regression", "Multiple Linear Regression", "Logistic Regression"]
    )

    target = st.selectbox("Select Target Variable", df.columns)

    features = st.multiselect(
        "Select Feature Variables",
        [col for col in df.columns if col != target]
    )
    if st.button("Run Model"):

        X = df[features]
        y = df[target]

        X = sm.add_constant(X)

        if model_type == "Logistic Regression":

        # Ensure binary target
            if len(y.unique()) != 2:
                st.error("Target must be binary (0/1) for Logistic Regression")
                st.stop()

            model = sm.Logit(y, X).fit()
            preds_prob = model.predict(X)
            preds = (preds_prob > 0.5).astype(int)

            params = model.params
            y_data = y

            st.session_state.result = {
                "type": "logistic",   # ✅ IMPORTANT
                "coef": params,
                "y_true": y_data,
                "y_pred": preds,
                "accuracy": accuracy_score(y, preds),
                "precision": precision_score(y, preds),
                "recall": recall_score(y, preds),
                 "f1": f1_score(y, preds)
            }

        else:
            model = sm.OLS(y, X).fit()
            preds = model.predict(X)
            params = model.params
            y_data = y

            equation = f"{target} = "

            for name in params.index:
                coef = round(params[name], 4)

                if name == "const":
                    equation += f"{coef} "
                else:
                    equation += f"+ ({coef} × {name}) "

            st.session_state.result = {
                "equation": equation,
                "r2": model.rsquared,
                "adj_r2": model.rsquared_adj,
                "coef": model.params,
                "pvalues": model.pvalues,
                "tvalues": model.tvalues,
                "stderr": model.bse,
                "fvalue": model.fvalue,
                "f_pvalue": model.f_pvalue,
                "n": len(y_data),
                "y_true": y_data,
                "y_pred": preds
            } 

# ---------------- RESULTS ----------------
if page == "Results" and st.session_state.result:

    res = st.session_state.result

    st.subheader("Results")

    # ==============================
    # 🟢 LOGISTIC REGRESSION RESULT
    # ==============================
    if res.get("type") == "logistic":

        st.markdown(f"""
        <div class="card-section blue-card">
            <h3>📊 Logistic Regression Metrics</h3>
            <p><b>Accuracy:</b> {round(res["accuracy"], 4)}</p>
            <p><b>Precision:</b> {round(res["precision"], 4)}</p>
            <p><b>Recall:</b> {round(res["recall"], 4)}</p>
            <p><b>F1 Score:</b> {round(res["f1"], 4)}</p>
        </div>
        """, unsafe_allow_html=True)

        # Confusion Matrix
        cm = confusion_matrix(res["y_true"], res["y_pred"])

        cm_df = pd.DataFrame(cm,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"]
        )

        st.markdown(f"""
        <div class="card-section orange-card">
            <h3>📋 Confusion Matrix</h3>
            {cm_df.to_html()}
        </div>
        """, unsafe_allow_html=True)

        # ROC AUC
        try:
            auc = roc_auc_score(res["y_true"], res["y_pred"])
            st.markdown(f"""
            <div class="card-section green-card">
                <h3>📈 ROC AUC Score</h3>
                <p>{round(auc, 4)}</p>
            </div>
            """, unsafe_allow_html=True)
        except:
            pass

    # ==============================
    # 🔵 LINEAR REGRESSION RESULT
    # ==============================
    else:

        st.markdown(f"""<div class="card-section blue-card">
        <h3>📌 Regression Equation</h3>
        <p>{res["equation"]}</p>
        </div>""", unsafe_allow_html=True)

        st.write("R²:", res["r2"])
        st.json(res["coef"].to_dict())

        stats_df = pd.DataFrame({
            "Metric": ["Multiple R", "R Square", "Adjusted R Square", "Standard Error", "Observations"],
            "Value": [
                round(res["r2"]**0.5, 6),
                round(res["r2"], 6),
                round(res["adj_r2"], 6),
                round((res["y_true"] - res["y_pred"]).std(), 4),
                res["n"]
            ]
        })

        st.markdown(f"""
        <div class="card-section green-card">
            <h3>📊 Regression Statistics</h3>
            {stats_df.to_html(index=False)}
        </div>
        """, unsafe_allow_html=True)

        ss_total = ((res["y_true"] - res["y_true"].mean())**2).sum()
        ss_res = ((res["y_true"] - res["y_pred"])**2).sum()
        ss_reg = ss_total - ss_res

        df_reg = len(res["coef"]) - 1
        df_res = res["n"] - len(res["coef"])

        anova_df = pd.DataFrame({
            "Source": ["Regression", "Residual", "Total"],
            "df": [df_reg, df_res, res["n"]-1],
            "SS": [ss_reg, ss_res, ss_total],
            "MS": [ss_reg/df_reg, ss_res/df_res, ""],
            "F": [res["fvalue"], "", ""],
            "Significance F": [res["f_pvalue"], "", ""]
        })

        st.markdown(f"""
        <div class="card-section orange-card">
            <h3>📋 ANOVA</h3>
            {anova_df.to_html(index=False)}
        </div>
        """, unsafe_allow_html=True)

        coef_df = pd.DataFrame({
            "Variable": res["coef"].index,
            "Coefficient": res["coef"].values,
            "Std Error": res["stderr"].values,
            "t Stat": res["tvalues"].values,
            "P-value": res["pvalues"].values
        })
        def highlight_p(val):
            return "color: green" if val < 0.05 else "color: red"

        st.dataframe(
            coef_df.style.map(highlight_p, subset=["P-value"]),
            use_container_width=True
        )
        st.markdown(f"""
        <div class="card-section white-card">
            <h3>📑 Coefficients</h3>
            {coef_df.to_html(index=False)}
        </div>
        """, unsafe_allow_html=True)

    
        st.subheader("📊 Diagnostic Plots")

        # Residual Histogram
        residuals = res["y_true"] - res["y_pred"]

        fig = px.histogram(residuals, nbins=50, title="Residual Histogram")
        st.plotly_chart(fig, use_container_width=True)

# Coefficient Importance
        coef_vals = res["coef"].drop("const", errors="ignore")

        fig2 = px.bar(
            x=coef_vals.index,
            y=coef_vals.values,
            title="Coefficient Importance"
        )

        st.plotly_chart(fig2, use_container_width=True)
        # 📈 Actual vs Predicted Plot
        st.subheader("📈 Actual vs Predicted")

        fig = px.scatter(
            x=res["y_true"],
            y=res["y_pred"],
            labels={"x": "Actual", "y": "Predicted"}
        )

        st.plotly_chart(fig, use_container_width=True)
  
    # 📉 Residual Plot
        st.subheader("📉 Residual Plot")
 
        residuals = res["y_true"] - res["y_pred"]

        fig2 = px.scatter(
            x=res["y_pred"],
            y=residuals,
            labels={"x": "Predicted", "y": "Residuals"}
        ) 

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🤖 AI Chat Assistant")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask something about your model...")

    if st.button("Send") and user_input:
    
        # Save user message
        st.session_state.chat_history.append(("You", user_input))

    # Get AI response
        reply = chatbot_response(user_input, res)

    # Save AI response
        st.session_state.chat_history.append(("AI", reply))


# Display chat
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 AI:** {msg}")