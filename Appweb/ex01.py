
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# 1. Configuration
# -----------------------------
st.set_page_config(
    page_title="Analyse automobile",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Analyse et prédiction de la consommation")
st.write(
    "Explorez les données automobiles et estimez "
    "la consommation d'un véhicule avec le Machine Learning."
)

URL = (
    "https://raw.githubusercontent.com/"
    "mwaskom/seaborn-data/master/mpg.csv"
)

# -----------------------------
# 2. Chargement des données
# -----------------------------
@st.cache_data
def charger_donnees():
    df = pd.read_csv(URL)
    df["horsepower"] = pd.to_numeric(
        df["horsepower"], errors="coerce"
    )
    df = df.dropna(
        subset=[
            "mpg", "cylinders", "displacement",
            "horsepower", "weight", "acceleration",
            "model_year", "origin"
        ]
    )
    return df

try:
    df = charger_donnees()
except Exception as e:
    st.error(f"Impossible de charger le CSV : {e}")
    st.stop()

# -----------------------------
# 3. Navigation
# -----------------------------
page = st.sidebar.radio(
    "Choisir une fonctionnalité",
    [
        "Analyse descriptive",
        "Prédiction avec Machine Learning"
    ]
)

# ==================================================
# PARTIE 1 : ANALYSE DESCRIPTIVE SANS ML
# ==================================================
if page == "Analyse descriptive":

    st.header("📊 Exploration des données")

    # Filtres
    st.sidebar.subheader("Filtres")

    origines = st.sidebar.multiselect(
        "Origine",
        options=sorted(df["origin"].unique()),
        default=sorted(df["origin"].unique())
    )

    annees = sorted(df["model_year"].unique())

    annee_min, annee_max = st.sidebar.select_slider(
        "Année-modèle",
        options=annees,
        value=(annees[0], annees[-1])
    )

    cylindres = st.sidebar.multiselect(
        "Nombre de cylindres",
        options=sorted(df["cylinders"].unique()),
        default=sorted(df["cylinders"].unique())
    )

    # Application des filtres
    filtre = df[
        df["origin"].isin(origines)
        & df["model_year"].between(annee_min, annee_max)
        & df["cylinders"].isin(cylindres)
    ]

    if filtre.empty:
        st.warning("Aucun véhicule ne correspond aux filtres.")
        st.stop()

    # Indicateurs
    col1, col2, col3 = st.columns(3)

    col1.metric("Nombre de véhicules", len(filtre))
    col2.metric(
        "Puissance moyenne",
        f"{filtre['horsepower'].mean():.1f} ch"
    )
    col3.metric(
        "Consommation moyenne",
        f"{filtre['mpg'].mean():.2f} MPG"
    )

    st.caption(
        "MPG = miles per gallon. Une valeur plus élevée "
        "indique une meilleure efficacité énergétique."
    )

    # Tableau
    st.subheader("Données filtrées")
    st.dataframe(filtre, use_container_width=True)

    # Graphiques
    gauche, droite = st.columns(2)

    with gauche:
        st.subheader("Poids et consommation")

        fig1 = px.scatter(
            filtre,
            x="weight",
            y="mpg",
            color="origin",
            hover_name="name",
            labels={
                "weight": "Poids (livres)",
                "mpg": "Consommation (MPG)",
                "origin": "Origine"
            },
            title="Relation entre poids et MPG"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with droite:
        st.subheader("Comparaison par origine")

        comparaison = (
            filtre.groupby("origin", as_index=False)
            .agg(
                consommation_moyenne=("mpg", "mean"),
                puissance_moyenne=("horsepower", "mean"),
                nombre_vehicules=("mpg", "size")
            )
        )

        fig2 = px.bar(
            comparaison,
            x="origin",
            y="consommation_moyenne",
            color="origin",
            labels={
                "origin": "Origine",
                "consommation_moyenne": "MPG moyen"
            },
            title="Consommation moyenne par origine"
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(
        filtre[
            [
                "mpg", "cylinders", "horsepower",
                "weight", "acceleration", "model_year"
            ]
        ].describe().round(2),
        use_container_width=True
    )

# ==================================================
# PARTIE 2 : PREDICTION AVEC MACHINE LEARNING
# ==================================================
else:

    st.header("🤖 Prédiction avec Random Forest")

    st.write(
        "Saisissez les caractéristiques d'un véhicule "
        "pour estimer son efficacité énergétique."
    )

    # Variables explicatives et cible
    caracteristiques = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin"
    ]

    X = df[caracteristiques]
    y = df["mpg"]

    numeriques = [
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year"
    ]

    # Encodage de l'origine
    preprocesseur = ColumnTransformer(
        transformers=[
            (
                "origine",
                OneHotEncoder(handle_unknown="ignore"),
                ["origin"]
            )
        ],
        remainder="passthrough"
    )

    # Pipeline de Machine Learning
    modele = Pipeline(
        steps=[
            ("pretraitement", preprocesseur),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                )
            )
        ]
    )

    # Séparation entraînement / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Entraînement et évaluation
    modele.fit(X_train, y_train)
    predictions_test = modele.predict(X_test)

    mae = mean_absolute_error(y_test, predictions_test)
    r2 = r2_score(y_test, predictions_test)

    st.subheader("Évaluation du modèle")

    c1, c2 = st.columns(2)
    c1.metric("Erreur absolue moyenne (MAE)", f"{mae:.2f} MPG")
    c2.metric("Coefficient R²", f"{r2:.3f}")

    st.caption(
        "Évaluation sur un jeu de test distinct des données "
        "utilisées pour entraîner le modèle."
    )

    # Formulaire de saisie
    st.subheader("Caractéristiques du véhicule")

    with st.form("form_prediction"):

        c1, c2 = st.columns(2)

        with c1:
            origine = st.selectbox(
                "Origine",
                sorted(df["origin"].unique())
            )

            cylindres = st.selectbox(
                "Nombre de cylindres",
                sorted(df["cylinders"].unique())
            )

            cylindree = st.number_input(
                "Cylindrée (cubic inches)",
                min_value=60.0,
                max_value=500.0,
                value=200.0,
                step=5.0
            )

            puissance = st.number_input(
                "Puissance (horsepower)",
                min_value=40.0,
                max_value=250.0,
                value=100.0,
                step=5.0
            )

        with c2:
            poids = st.number_input(
                "Poids (livres)",
                min_value=1500.0,
                max_value=5500.0,
                value=3000.0,
                step=100.0
            )

            acceleration = st.number_input(
                "Accélération (secondes)",
                min_value=8.0,
                max_value=30.0,
                value=15.0,
                step=0.5
            )

            annee = st.selectbox(
                "Année-modèle",
                sorted(df["model_year"].unique())
            )

        soumettre = st.form_submit_button(
            "Prédire la consommation"
        )

    if soumettre:

        nouveau = pd.DataFrame(
            [[
                cylindres,
                cylindree,
                puissance,
                poids,
                acceleration,
                annee,
                origine
            ]],
            columns=caracteristiques
        )

        resultat = modele.predict(nouveau)[0]

        st.success(
            f"Consommation estimée : {resultat:.2f} MPG"
        )

        # Conversion indicative en litres / 100 km
        l100km = 235.215 / resultat

        st.metric(
            "Équivalent approximatif",
            f"{l100km:.2f} L/100 km"
        )

        st.caption(
            "Cette estimation est produite par un modèle "
            "entraîné sur des véhicules historiques. "
            "Elle ne garantit pas la consommation réelle."
        )

st.sidebar.divider()
st.sidebar.caption("Projet Python • Streamlit • Pandas • Scikit-learn")
