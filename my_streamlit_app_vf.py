import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Page config
st.set_page_config(
    page_title="Caixa Banco • Churn Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Neon Dark
st.markdown("""
<style>
:root{
  --bg:#070A10; --panel:rgba(255,255,255,.04); --stroke:rgba(255,255,255,.08);
  --text:rgba(255,255,255,.9); --muted:rgba(255,255,255,.55);
  --neon:#37F5C8; --violet:#7C5CFF; --danger:#FF4D6D; --ok:#3DFF8A;
}
.neon-title-center {
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 0.06em;
    margin: 35px 0 20px 0;
    color: var(--neon);
    text-shadow:
        0 0 6px rgba(55,245,200,0.7),
        0 0 14px rgba(55,245,200,0.45),
        0 0 28px rgba(55,245,200,0.25);
}
.normal-title-center {
    text-align: center;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 0.06em;
    margin: 35px 0 20px 0;
    color: #FFFFFF;
    text-shadow:
        0 0 6px rgba(255,255,255,0.6),
        0 0 14px rgba(255,255,255,0.35);
}
.neon-title-left {
    text-align: left;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 0.04em;
    margin: 30px 0 15px 0;
    color: var(--neon);
    text-shadow:
        0 0 6px rgba(55,245,200,0.6),
        0 0 12px rgba(55,245,200,0.35);
}
html, body, [class*="css"]{
  background: radial-gradient(1200px 600px at 20% 10%, rgba(55,245,200,.12), transparent 60%),
              radial-gradient(1000px 500px at 80% 20%, rgba(124,92,255,.12), transparent 55%),
              var(--bg)!important;
  color:var(--text);
}
.panel{
  background:linear-gradient(180deg,var(--panel),rgba(255,255,255,.02));
  border:1px solid var(--stroke); border-radius:18px;
  padding:18px; box-shadow:0 14px 50px rgba(0,0,0,.35);
}
.kpi .label{font-size:12px;color:var(--muted);text-transform:uppercase}
.kpi .value{font-size:42px;font-weight:800}
.stButton>button{
  border-radius:14px; background:rgba(55,245,200,.15);
  border:1px solid rgba(55,245,200,.4); font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["✨ Prédiction client", "📊 Suivi des données (EDA)"]
)

st.sidebar.markdown("### Modèle & décision")
model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["XGBoost", "RandomForest", "Bagging"]
)

threshold = st.sidebar.slider(
    "Seuil de décision (churn)",
    0.30, 0.70, 0.50, 0.05
)

# Model training
@st.cache_resource
def train_model(model_choice):
    df = pd.read_csv("Caixa Banco.csv")
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    numeric_features = [
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary"
    ]
    categorical_features = ["Geography", "Gender"]

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ])

    if model_choice == "XGBoost":
        model = XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=42
        )
    elif model_choice == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        )
    else:
        model = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=100, random_state=42
        )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline.fit(X_train, y_train)
    return pipeline

pipeline = train_model(model_choice)

# SHAP
@st.cache_resource
def init_shap(_pipeline):
    model = _pipeline.named_steps["model"]
    return shap.TreeExplainer(model)

explainer = init_shap(pipeline)

def compute_shap(pipeline, explainer, input_df):
    X_t = pipeline.named_steps["preprocess"].transform(input_df)
    values = explainer.shap_values(X_t)

    num = pipeline.named_steps["preprocess"].transformers_[0][2]
    cat = pipeline.named_steps["preprocess"].transformers_[1][1].get_feature_names_out(
        pipeline.named_steps["preprocess"].transformers_[1][2]
    )
    names = list(num) + list(cat)

    df = pd.DataFrame({"Feature": names, "Impact": values[0]})
    df["abs"] = df["Impact"].abs()
    return df.sort_values("abs", ascending=False).head(5)

def get_feature_importance(pipeline, top_n=10):
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]

    num_features = preprocess.transformers_[0][2]
    cat_features = preprocess.transformers_[1][1].get_feature_names_out(
        preprocess.transformers_[1][2]
    )
    feature_names = list(num_features) + list(cat_features)

    # Cas 1 : modèle avec feature_importances_
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # Cas 2 : Bagging → moyenne des arbres
    elif hasattr(model, "estimators_"):
        importances = np.mean(
            [est.feature_importances_ for est in model.estimators_],
            axis=0
        )

    else:
        raise ValueError("Ce modèle ne supporte pas l'importance des variables.")

    df_importance = (
        pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        })
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    return df_importance

# PAGE 1 — PREDICTION
if page == "✨ Prédiction client":

    st.markdown("<div class='neon-title-left'>✨ Prédiction du churn client</div>",unsafe_allow_html=True)

    st.sidebar.markdown("### Profil client")
    credit_score = st.sidebar.slider("Score de crédit", 350, 850, 600)
    age = st.sidebar.slider("Âge", 18, 90, 40)
    tenure = st.sidebar.slider("Ancienneté", 0, 10, 3)
    balance = st.sidebar.number_input("Solde (€)", 0.0, 300000.0, 50000.0, 1000.0)
    num_products = st.sidebar.selectbox("Produits", [1, 2, 3, 4])
    has_card = st.sidebar.toggle("Carte de crédit", True)
    active = st.sidebar.toggle("Client actif", True)
    salary = st.sidebar.number_input("Salaire estimé", 0.0, 200000.0, 50000.0, 1000.0)
    country = st.sidebar.selectbox("Pays", ["France", "Germany", "Spain"])
    gender = st.sidebar.selectbox("Genre", ["Male", "Female"])

    input_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_card),
        "IsActiveMember": int(active),
        "EstimatedSalary": salary,
        "Geography": country,
        "Gender": gender
    }])

    if "proba" not in st.session_state:
        st.session_state.proba = None

    if st.button("Analyser le risque", use_container_width=True):
        st.session_state.proba = float(pipeline.predict_proba(input_df)[0, 1])

    if st.session_state.proba is not None:
        k1, k2, k3 = st.columns(3)
        # KPI 1 — Probabilité
        with k1:
            st.markdown(
                f"""
                <div class="panel glow kpi">
                    <div class="label">Probabilité de churn</div>
                    <div class="value">{st.session_state.proba:.0%}</div>
                    <div class="hint">Sortie du modèle</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        # KPI 2 — Seuil
        with k2:
            st.markdown(
                f"""
                <div class="panel glow kpi">
                    <div class="label">Seuil de décision</div>
                    <div class="value">{threshold:.2f}</div>
                    <div class="hint">Paramètre métier</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        # KPI 3 — Décision
        decision = "Risque élevé" if st.session_state.proba >= threshold else "Risque faible"
        color = "#FF4D6D" if decision == "Risque élevé" else "#3DFF8A"

        with k3:
            st.markdown(
                f"""
                <div class="panel glow kpi">
                    <div class="label">Décision</div>
                    <div class="value" style="color:{color}">{decision}</div>
                    <div class="hint">Comparaison proba / seuil</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")
        st.markdown("<div class='neon-title-left'> Pourquoi ce score ?</div>",unsafe_allow_html=True)
        shap_df = compute_shap(pipeline, explainer, input_df)
        for _, r in shap_df.iterrows():
            col = "#FF4D6D" if r["Impact"] > 0 else "#3DFF8A"
            sign = "+" if r["Impact"] > 0 else ""
            st.markdown(f"**{r['Feature']}** : <span style='color:{col}'>{sign}{r['Impact']:.3f}</span>",
                        unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='neon-title-center'> Facteurs globaux</div>",unsafe_allow_html=True)
    fi = get_feature_importance(pipeline)
    fig = px.bar(fi[::-1], x="Importance", y="Feature",
                 orientation="h", color="Importance",
                 color_continuous_scale=["#37F5C8", "#7C5CFF"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)

# PAGE 2 — EDA
else:
    df = pd.read_csv("Caixa Banco.csv")

    st.markdown("<div class='neon-title-left'>📊 Suivi des données – EDA - Churn Intelligence Center</div>",unsafe_allow_html=True)
    st.caption("Vue analytique globale pour comprendre les facteurs de churn")

    # 1️. HERO KPI SECTION
    churn_rate = df["Exited"].mean()
    total_clients = len(df)

    worst_country = (
        df.groupby("Geography")["Exited"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )

    critical_segment = "Clients inactifs"

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="panel glow kpi">
            <div class="label">Taux de churn</div>
            <div class="value" style="color:#FF4D6D">{churn_rate:.1%}</div>
            <div class="hint">Vue globale</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="panel glow kpi">
            <div class="label">Clients analysés</div>
            <div class="value">{total_clients}</div>
            <div class="hint">Total du fichier source</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="panel glow kpi">
            <div class="label">Pays à risque</div>
            <div class="value">{worst_country}</div>
            <div class="hint">Churn le plus élevé</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="panel glow kpi">
            <div class="label">Segment critique</div>
            <div class="value" style="font-size:34px;">{critical_segment}</div>
            <div class="hint">Action prioritaire</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 2️. CHURN MAP — Heatmap Age × Produits
    st.markdown("<div class='neon-title-center'> Churn Map – Segments à risque</div>",unsafe_allow_html=True)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[18, 30, 40, 50, 60, 100],
        labels=["18–30", "30–40", "40–50", "50–60", "60+"]
    )

    heatmap_df = (
        df.groupby(["AgeGroup", "NumOfProducts"])["Exited"]
        .mean()
        .reset_index()
    )

    fig_heatmap = px.density_heatmap(
        heatmap_df,
        x="AgeGroup",
        y="NumOfProducts",
        z="Exited",
        color_continuous_scale=["#37F5C8", "#FF4D6D"],
        title="Taux de churn par âge et nombre de produits"
    )

    fig_heatmap.update_layout(
        title_x=0.33,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # 3️. STORYTELLING — QUI SONT LES CHURNERS ?
    st.markdown("<div class='neon-title-center'> Qui sont les clients churners ?</div>",unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='normal-title-center'> Âge</div>",unsafe_allow_html=True)
        fig_age = px.box(
            df,
            x="Exited",
            y="Age",
            color="Exited",
            color_discrete_sequence=["#37F5C8", "#FF4D6D"]
        )
        fig_age.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with c2:
        st.markdown("<div class='normal-title-center'> Activité client</div>",unsafe_allow_html=True)
        fig_active = px.bar(
            df.groupby("IsActiveMember")["Exited"].mean().reset_index(),
            x="IsActiveMember",
            y="Exited",
            color_discrete_sequence=["#7C5CFF"]
        )
        fig_active.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig_active, use_container_width=True)

    st.divider()

    # 4️. CAMEMBERT CHURN / NON-CHURN
    st.markdown("<div class='neon-title-center'> Répartition globale du churn</div>",unsafe_allow_html=True)

    churn_counts = df["Exited"].value_counts().rename({0: "Non churn", 1: "Churn"})

    fig_pie = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        hole=0.45,
        color=churn_counts.index,
        color_discrete_map={
            "Non churn": "#37F5C8",
            "Churn": "#FF4D6D"
        }
    )

    fig_pie.update_traces(
        textinfo="percent",
        textposition="outside",
        pull=[0.05, 0.12],
        marker=dict(line=dict(color="#070A10", width=2)),
        showlegend=True
    )

    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        uniformtext_minsize=14,
        uniformtext_mode="hide"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # 5. Histogramme GENRE
    st.markdown("<div class='neon-title-center'> Taux de churn par genre</div>",unsafe_allow_html=True)

    gender_churn = (
        df.groupby("Gender")["Exited"]
        .mean()
        .reset_index()
    )

    fig_gender = px.bar(
        gender_churn,
        x="Gender",
        y="Exited",
        text=gender_churn["Exited"].map(lambda x: f"{x:.1%}")
    )

    fig_gender.update_traces(
        marker_color=["#FF5DA2", "#3A7CFF"],
        text=gender_churn["Exited"].map(lambda x: f"{x:.1%}"),
        textposition="outside",
        textfont=dict(color="white", size=14),
        width=0.35
    )

    fig_gender.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=["Female", "Male"]
        ),
        yaxis_title="Taux de churn",
        xaxis_title="Genre",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        showlegend=False
    )

    st.plotly_chart(fig_gender, use_container_width=True)

    st.divider()

    # 6. Histogramme - Géographie
    st.markdown("<div class='neon-title-center'> Taux de churn par pays</div>",unsafe_allow_html=True)

    geo_churn = (
        df.groupby("Geography")["Exited"]
        .mean()
        .reset_index()
    )

    fig_geo = px.bar(
        geo_churn,
        x="Geography",
        y="Exited",
        text=geo_churn["Exited"].map(lambda x: f"{x:.1%}")
    )

    fig_geo.update_traces(
        marker_color=["#3A7CFF", "#FF4D4D", "#FFD43B"],
        text=geo_churn["Exited"].map(lambda x: f"{x:.1%}"),
        textposition="outside",
        textfont=dict(color="white", size=14),
        width=0.35
    )

    fig_geo.update_layout(
        xaxis=dict(
            categoryorder="array",
            categoryarray=["France", "Germany", "Spain"]
        ),
        yaxis_title="Taux de churn",
        xaxis_title="Pays",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        showlegend=False
    )

    st.plotly_chart(fig_geo, use_container_width=True)

    st.divider()

    # 7. SYNTHÈSE MÉTIER
    st.markdown("<div class='neon-title-left'> Synthèse - les éléments clés</div>",unsafe_allow_html=True)

    st.markdown("""
    - Les **clients inactifs** présentent un taux de churn significativement plus élevé
    - Le churn augmente à partir de **40 ans**, surtout avec peu de produits
    - **L’Allemagne** est le pays le plus à risque
    - Les clients avec **1 seul produit** sont les plus fragiles
    """)