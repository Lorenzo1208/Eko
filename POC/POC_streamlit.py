import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Eko - Surveillance des Rivières d'Occitanie",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fonction pour charger les données des mesures hydrométriques
@st.cache_data
def load_hydro_data(file_path="mesures_hydro_occitanie_2020_2025_20250520_100820.csv", sample_size=1000):
    try:
        # Déterminer le nombre de lignes total
        total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
        
        # Calculer un pas d'échantillonnage
        skip_rows = max(1, total_lines // sample_size)
        
        # Charger un échantillon du fichier
        df = pd.read_csv(file_path, 
                         skiprows=lambda i: i > 0 and i % skip_rows != 0,
                         encoding='utf-8')
        
        # Traitement des dates...
        date_column = 'date_obs_elab' if 'date_obs_elab' in df.columns else 'date_obs'
        df['date'] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Extraire des informations temporelles
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return None

# Fonction pour créer un DataFrame des stations uniques avec leurs coordonnées
@st.cache_data
def extract_stations_info(data_df):
    if data_df is None:
        return None
    
    try:
        # Sélectionner les colonnes pertinentes et supprimer les doublons
        station_cols = ['code_station_id', 'nom_station', 'longitude', 'latitude']
        
        # Vérifier si toutes les colonnes existent
        if not all(col in data_df.columns for col in station_cols):
            # Pour les colonnes manquantes, créer des mappings par défaut
            if 'code_station_id' not in data_df.columns and 'code_station' in data_df.columns:
                data_df['code_station_id'] = data_df['code_station']
                
            # Si des coordonnées sont manquantes, générer des coordonnées fictives centrées sur l'Occitanie
            if 'longitude' not in data_df.columns:
                data_df['longitude'] = 1.444 + np.random.normal(0, 0.5, len(data_df))
            if 'latitude' not in data_df.columns:
                data_df['latitude'] = 43.6 + np.random.normal(0, 0.3, len(data_df))
        
        # Sélectionner les colonnes disponibles
        available_cols = [col for col in station_cols if col in data_df.columns]
        
        # Créer le DataFrame des stations
        stations_df = data_df[available_cols].drop_duplicates().reset_index(drop=True)
        
        # Ajouter des informations sur le cours d'eau si disponibles
        if 'libelle_cours_eau' in data_df.columns:
            stations_df = pd.merge(
                stations_df,
                data_df[['code_station_id', 'libelle_cours_eau']].drop_duplicates(),
                on='code_station_id',
                how='left'
            )
        
        return stations_df
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des informations des stations: {str(e)}")
        return None

# Fonction pour calculer les seuils statistiques par station
@st.cache_data
def calculate_thresholds(df):
    if df is None or len(df) == 0:
        return None
        
    try:
        # Identifier la colonne de débit
        flow_column = 'resultat_obs_elab' if 'resultat_obs_elab' in df.columns else 'resultat_obs'
        
        # Grouper par station
        grouped = df.groupby('code_station_id')
        
        # Calculer les seuils
        thresholds = grouped.agg(
            debit_min=(flow_column, 'min'),
            debit_max=(flow_column, 'max'),
            debit_moyen=(flow_column, 'mean'),
            seuil_vigilance=(flow_column, lambda x: np.percentile(x, 75)),
            seuil_alerte=(flow_column, lambda x: np.percentile(x, 90)),
            seuil_alerte_renforcee=(flow_column, lambda x: np.percentile(x, 95)),
            seuil_crise=(flow_column, lambda x: np.percentile(x, 98))
        ).reset_index()
        
        return thresholds
    except Exception as e:
        st.error(f"Erreur lors du calcul des seuils: {str(e)}")
        return None

# Fonction pour calculer les seuils mensuels
@st.cache_data
def calculate_monthly_thresholds(df):
    if df is None or len(df) == 0:
        return None
        
    try:
        # Identifier la colonne de débit
        flow_column = 'resultat_obs_elab' if 'resultat_obs_elab' in df.columns else 'resultat_obs'
        
        # Grouper par station et mois
        grouped = df.groupby(['code_station_id', 'month'])
        
        # Calculer les seuils
        thresholds = grouped.agg(
            debit_min=(flow_column, 'min'),
            debit_max=(flow_column, 'max'),
            debit_moyen=(flow_column, 'mean'),
            seuil_vigilance=(flow_column, lambda x: np.percentile(x, 75)),
            seuil_alerte=(flow_column, lambda x: np.percentile(x, 90)),
            seuil_alerte_renforcee=(flow_column, lambda x: np.percentile(x, 95)),
            seuil_crise=(flow_column, lambda x: np.percentile(x, 98))
        ).reset_index()
        
        return thresholds
    except Exception as e:
        st.error(f"Erreur lors du calcul des seuils mensuels: {str(e)}")
        return None

# Fonction pour créer un modèle de prédiction (RandomForest ou Prophet)
@st.cache_resource
def create_prediction_model(df, station_id, method="randomforest"):
    if df is None or len(df) == 0:
        return None
        
    try:
        # Filtrer les données pour la station sélectionnée
        station_data = df[df['code_station_id'] == station_id]
        
        # Vérifier s'il y a suffisamment de données
        if len(station_data) < 10:
            st.warning(f"Pas assez de données pour la station {station_id}. Le modèle peut être imprécis.")
            if len(station_data) < 5:
                return None
        
        # Choisir le bon modèle en fonction de la méthode
        if method.lower() == "prophet":
            # Préparer les données pour Prophet
            flow_column = 'resultat_obs_elab' if 'resultat_obs_elab' in station_data.columns else 'resultat_obs'
            prophet_df = pd.DataFrame({
                'ds': station_data['date'],
                'y': station_data[flow_column]
            })
            
            # Créer et entraîner le modèle
            model = Prophet(daily_seasonality=True)
            model.fit(prophet_df)
            return {"type": "prophet", "model": model}
            
        else:  # RandomForest par défaut
            # Préparer les données pour RandomForest
            flow_column = 'resultat_obs_elab' if 'resultat_obs_elab' in station_data.columns else 'resultat_obs'
            
            # Caractéristiques de base pour le modèle (indépendantes de la saison)
            features = ['month', 'day_of_year']
            
            # Ajouter la saison encodée si disponible
            saison_encoder = None
            if 'saison' in station_data.columns:
                # Encoder la saison
                le = LabelEncoder()
                station_data['saison_encoded'] = le.fit_transform(station_data['saison'])
                features.append('saison_encoded')
                saison_encoder = le
            
            # S'assurer que toutes les features existent
            X = station_data[features].copy()  # Créer une copie pour éviter les warnings
            y = station_data[flow_column]
            
            # Diviser en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entraîner le modèle
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Stocker les informations importantes pour les prédictions
            model_info = {
                "type": "randomforest",
                "model": rf_model,
                "features": features,
                "saison_encoder": saison_encoder,
                "has_saison_feature": 'saison' in station_data.columns
            }
            
            return model_info
            
    except Exception as e:
        st.error(f"Erreur lors de la création du modèle de prédiction: {str(e)}")
        return None

# Fonction pour faire des prédictions avec le modèle
def make_predictions(model_info, df, station_id, days=30):
    if model_info is None:
        return None
        
    try:
        # Récupérer les informations de la station
        station_data = df[df['code_station_id'] == station_id]
        
        # Déterminer la dernière date disponible
        last_date = station_data['date'].max()
        
        # Générer les dates futures
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        
        # Faire les prédictions en fonction du type de modèle
        if model_info["type"] == "prophet":
            # Créer un DataFrame avec les dates futures pour Prophet
            future = pd.DataFrame({"ds": pd.date_range(start=last_date, periods=days+1)[1:]})
            
            # Faire les prédictions
            forecast = model_info["model"].predict(future)
            
            # Préparer le résultat
            result = pd.DataFrame({
                "date": forecast["ds"],
                "prediction": forecast["yhat"],
                "lower_bound": forecast["yhat_lower"],
                "upper_bound": forecast["yhat_upper"]
            })
            
        else:  # RandomForest
            # Créer les caractéristiques pour les dates futures
            future_features = []
            for date in future_dates:
                features_dict = {
                    "month": date.month,
                    "day_of_year": date.timetuple().tm_yday
                }
                
                # Ajouter la saison si nécessaire
                if model_info.get("has_saison_feature", False) and model_info.get("saison_encoder") is not None:
                    # Déterminer la saison en fonction du mois
                    if 3 <= date.month <= 5:
                        saison = "Printemps"
                    elif 6 <= date.month <= 8:
                        saison = "Été"
                    elif 9 <= date.month <= 11:
                        saison = "Automne"
                    else:
                        saison = "Hiver"
                    
                    features_dict["saison_encoded"] = model_info["saison_encoder"].transform([saison])[0]
                    
                future_features.append(features_dict)
            
            # Créer un DataFrame avec les caractéristiques futures
            future_df = pd.DataFrame(future_features)
            
            # S'assurer que toutes les features utilisées lors de l'entraînement sont présentes
            for feature in model_info["features"]:
                if feature not in future_df.columns:
                    st.warning(f"Feature manquante: {feature}. Le modèle pourrait être imprécis.")
                    # Si c'est la saison qui manque, on pourrait créer une colonne factice
                    if feature == 'saison_encoded' and 'month' in future_df.columns:
                        # Créer une saison encodée basée sur le mois
                        future_df['saison_encoded'] = future_df['month'].apply(
                            lambda m: 0 if m in [12, 1, 2] else  # Hiver
                                    1 if m in [3, 4, 5] else     # Printemps
                                    2 if m in [6, 7, 8] else     # Été
                                    3                            # Automne
                        )
            
            # Faire les prédictions en utilisant uniquement les features disponibles dans le modèle
            available_features = [f for f in model_info["features"] if f in future_df.columns]
            predictions = model_info["model"].predict(future_df[available_features])
            
            # Calculer des intervalles de confiance approximatifs
            std_dev = np.std(predictions) if len(predictions) > 1 else 0
            lower_bounds = predictions - 1.96 * std_dev
            upper_bounds = predictions + 1.96 * std_dev
            
            # Préparer le résultat
            result = pd.DataFrame({
                "date": future_dates,
                "prediction": predictions,
                "lower_bound": lower_bounds,
                "upper_bound": upper_bounds
            })
            
        return result
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        st.exception(e)  # Afficher la trace complète pour déboguer
        return None

# Fonction pour afficher une carte des stations avec PyDeck
def display_station_map(stations_df, thresholds_df):
    if stations_df is None or len(stations_df) == 0:
        st.error("Aucune donnée de station disponible pour la carte.")
        return None
        
    try:
        # Fusionner les données des stations avec les seuils
        if thresholds_df is not None:
            map_data = pd.merge(
                stations_df, 
                thresholds_df, 
                left_on='code_station_id',
                right_on='code_station_id', 
                how='left'
            )
        else:
            map_data = stations_df.copy()
        
        # Afficher les valeurs numériques correctement formatées
        for col in ['debit_moyen', 'seuil_alerte', 'seuil_crise']:
            if col in map_data.columns:
                map_data[f'{col}_fmt'] = map_data[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        
        # Créer une colonne pour la couleur en fonction du seuil
        def determine_color(row):
            if 'debit_moyen' not in row or pd.isna(row.get('debit_moyen')):
                return [150, 150, 150]  # Gris pour les stations sans données
            
            if 'seuil_crise' in row and not pd.isna(row.get('seuil_crise')):
                if row['debit_moyen'] > row['seuil_crise']:
                    return [255, 0, 0]  # Rouge pour stations au-dessus du seuil de crise
                elif row['debit_moyen'] > row['seuil_alerte']:
                    return [255, 165, 0]  # Orange pour stations au-dessus du seuil d'alerte
                else:
                    return [0, 100, 255]  # Bleu pour stations normales
            else:
                return [0, 100, 255]  # Bleu par défaut
        
        map_data['color'] = map_data.apply(determine_color, axis=1)
        
        # Créer la couche des stations - TAILLE RÉDUITE
        stations_layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position=['longitude', 'latitude'],
            get_color='color',
            get_radius=5000,  # Rayon réduit à 5 km (au lieu de 25 km)
            pickable=True,
            auto_highlight=True,
            opacity=0.8,
        )
        
        # Définir la vue initiale (centrée sur l'Occitanie)
        view_state = pdk.ViewState(
            longitude=1.4442,  # Coordonnées approximatives pour l'Occitanie
            latitude=43.6047,
            zoom=7,
            pitch=0,
        )
        
        # Créer le tooltip avec les valeurs formatées
        tooltip = {
            "html": "<b>Station:</b> {nom_station}<br>",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
        
        # Ajouter les informations de débit si disponibles
        if 'debit_moyen_fmt' in map_data.columns:
            tooltip["html"] += "<b>Débit moyen:</b> {debit_moyen_fmt} m³/s<br>"
        
        if 'seuil_alerte_fmt' in map_data.columns:
            tooltip["html"] += "<b>Seuil alerte:</b> {seuil_alerte_fmt} m³/s"
        
        # Créer la carte
        r = pdk.Deck(
            layers=[stations_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="light",
        )
        
        return r
    except Exception as e:
        st.error(f"Erreur lors de la création de la carte: {str(e)}")
        return None

def main():
    # Titre de l'application
    st.title("🌊 Eko - Surveillance des Rivières d'Occitanie")
    
    # Charger les données
    data = load_hydro_data()
    
    # Si aucun fichier par défaut n'est trouvé, permettre à l'utilisateur d'uploader son propre fichier
    if data is None:
        st.warning("Aucun fichier de données par défaut trouvé. Veuillez uploader votre CSV de données hydrométriques.")
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            # Convertir les dates et ajouter les colonnes temporelles
            date_column = 'date_obs_elab' if 'date_obs_elab' in data.columns else 'date_obs'
            if date_column in data.columns:
                data['date'] = pd.to_datetime(data[date_column])
                data['year'] = data['date'].dt.year
                data['month'] = data['date'].dt.month
                data['day'] = data['date'].dt.day
                data['day_of_year'] = data['date'].dt.dayofyear
    
    # Vérifier si des données sont disponibles
    if data is None or len(data) == 0:
        st.error("Aucune donnée disponible. Veuillez vérifier votre fichier CSV.")
        return
    
    # Extraire les informations des stations
    stations_info = extract_stations_info(data)
    
    # Calcul des seuils
    with st.spinner("Calcul des seuils d'alerte..."):
        thresholds_df = calculate_thresholds(data)
        monthly_thresholds_df = calculate_monthly_thresholds(data)
    
    # Afficher des informations sur les données chargées
    st.sidebar.subheader("Informations sur les données")
    st.sidebar.info(f"Nombre de stations: {data['code_station_id'].nunique()}")
    st.sidebar.info(f"Nombre total d'observations: {len(data)}")
    
    if 'date' in data.columns:
        st.sidebar.info(f"Période des données: du {data['date'].min().strftime('%d/%m/%Y')} au {data['date'].max().strftime('%d/%m/%Y')}")
    
    # Sidebar pour les contrôles
    st.sidebar.title("Contrôles")
    
    # 1. Sélection de la station
    station_options = data[['code_station_id', 'nom_station']].drop_duplicates()
    
    # Vérifier si station_options n'est pas vide
    if len(station_options) > 0:
        selected_station = st.sidebar.selectbox(
            "Sélectionner une station",
            station_options['code_station_id'].tolist(),
            format_func=lambda x: f"{x} - {station_options[station_options['code_station_id'] == x]['nom_station'].values[0]}"
        )
    else:
        st.error("Aucune station disponible. Vérifiez le format de vos données.")
        return
    
    # 2. Type d'affichage
    display_type = st.sidebar.radio(
        "Type d'affichage",
        ["Carte des stations", "Historique des débits", "Seuils d'alerte", "Prédictions"]
    )
    
    # 3. Options pour les prédictions
    if display_type == "Prédictions":
        prediction_days = st.sidebar.slider(
            "Nombre de jours de prédiction",
            min_value=7,
            max_value=90,
            value=30,
            step=1
        )
        
        prediction_method = st.sidebar.radio(
            "Méthode de prédiction",
            ["RandomForest", "Prophet"],
            help="RandomForest est généralement plus précis pour ce type de données. Prophet est meilleur pour les tendances saisonnières."
        )
    else:
        prediction_days = 30  # Valeur par défaut
        prediction_method = "RandomForest"  # Méthode par défaut
    
    # Affichage principal en fonction du type sélectionné
    if display_type == "Carte des stations":
        st.header("Carte des stations hydrométriques")
        
        # Colonne d'information
        st.markdown("""
        **Légende**:
        - 🔵 Bleu: Station avec débit normal
        - 🟠 Orange: Station au-dessus du seuil d'alerte
        - 🔴 Rouge: Station au-dessus du seuil de crise
        - ⚪ Gris: Station sans données de débit
        """)
        
        # Afficher la carte
        map_chart = display_station_map(stations_info, thresholds_df)
        if map_chart:
            st.pydeck_chart(map_chart)
        
        # Informations supplémentaires
        st.subheader("Informations sur les stations")
        
        # Sélectionner les colonnes pertinentes pour l'affichage
        display_columns = ['code_station_id', 'nom_station', 'longitude', 'latitude']
        
        if stations_info is not None:
            st.dataframe(stations_info[display_columns])
    
    elif display_type == "Historique des débits":
        st.header(f"Historique des débits pour la station {selected_station}")
        
        # Filtrer les données pour la station sélectionnée
        flow_column = 'resultat_obs_elab' if 'resultat_obs_elab' in data.columns else 'resultat_obs'
        station_data = data[data['code_station_id'] == selected_station].copy()
        station_data = station_data.sort_values('date')
        # Vérifier si des données sont disponibles pour cette station
        if len(station_data) == 0:
            st.warning(f"Aucune donnée de débit disponible pour la station {selected_station}.")
            return
        
        # Obtenir les informations de la station
        station_name = station_options[station_options['code_station_id'] == selected_station]['nom_station'].iloc[0]
        st.subheader(f"Station: {station_name}")
        
        # Créer un graphique des débits avec Plotly
        fig = px.scatter(
            station_data,
            x='date',
            y=flow_column,
            title=f"Débits (m³/s) - Station {selected_station}",
            labels={flow_column: "Débit (m³/s)", 'date': 'Date'},
            template="plotly_white"
        )
        
        # Ajouter une ligne de tendance
        fig.add_traces(
            px.line(station_data, x='date', y=flow_column).data[0]
        )
        
        # Ajouter les seuils d'alerte s'ils existent
        if thresholds_df is not None:
            threshold_data = thresholds_df[thresholds_df['code_station_id'] == selected_station]
            
            if len(threshold_data) > 0:
                threshold_data = threshold_data.iloc[0]
                fig.add_hline(y=threshold_data['seuil_vigilance'], line_dash="dash", line_color="yellow", annotation_text="Vigilance")
                fig.add_hline(y=threshold_data['seuil_alerte'], line_dash="dash", line_color="orange", annotation_text="Alerte")
                fig.add_hline(y=threshold_data['seuil_alerte_renforcee'], line_dash="dash", line_color="orangered", annotation_text="Alerte renforcée")
                fig.add_hline(y=threshold_data['seuil_crise'], line_dash="dash", line_color="red", annotation_text="Crise")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques des débits
        st.subheader("Statistiques des débits")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Débit minimum", f"{station_data[flow_column].min():.2f} m³/s")
        col2.metric("Débit moyen", f"{station_data[flow_column].mean():.2f} m³/s")
        col3.metric("Débit maximum", f"{station_data[flow_column].max():.2f} m³/s")
        col4.metric("Écart-type", f"{station_data[flow_column].std():.2f} m³/s")
        
        # Répartition saisonnière
        if 'saison' in station_data.columns:
            st.subheader("Répartition saisonnière")
            
            # Statistiques par saison
            seasonal_stats = station_data.groupby('saison')[flow_column].agg(['mean', 'min', 'max']).reset_index()
            
            # Ordonner les saisons correctement
            season_order = ['Hiver', 'Printemps', 'Été', 'Automne']
            seasonal_stats['saison'] = pd.Categorical(seasonal_stats['saison'], categories=season_order, ordered=True)
            seasonal_stats = seasonal_stats.sort_values('saison')
            
            # Créer un graphique des débits par saison
            fig_season = px.bar(
                seasonal_stats,
                x='saison',
                y='mean',
                title="Débit moyen par saison",
                labels={'mean': "Débit moyen (m³/s)", 'saison': 'Saison'},
                text_auto='.2f',
                color='saison',
                color_discrete_map={
                    'Hiver': 'royalblue',
                    'Printemps': 'mediumseagreen',
                    'Été': 'orange',
                    'Automne': 'brown'
                }
            )
            
            st.plotly_chart(fig_season, use_container_width=True)
    
    elif display_type == "Seuils d'alerte":
        st.header("Seuils d'alerte calculés")
        
        # 1. Seuils globaux
        st.subheader("Seuils globaux par station")
        if thresholds_df is not None:
            # Joindre les noms des stations pour une meilleure lisibilité
            display_thresholds = pd.merge(
                thresholds_df,
                station_options,
                on='code_station_id',
                how='left'
            )
            
            # Formater les colonnes numériques pour un affichage plus lisible
            for col in display_thresholds.columns:
                if col not in ['code_station_id', 'nom_station'] and pd.api.types.is_numeric_dtype(display_thresholds[col]):
                    display_thresholds[col] = display_thresholds[col].round(2)
            
            st.dataframe(display_thresholds)
        else:
            st.warning("Aucun seuil global calculé.")
        
        # 2. Seuils mensuels pour la station sélectionnée
        st.subheader(f"Seuils mensuels pour la station {selected_station}")
        
        if monthly_thresholds_df is not None:
            monthly_data = monthly_thresholds_df[monthly_thresholds_df['code_station_id'] == selected_station].copy()
            
            if len(monthly_data) == 0:
                st.warning(f"Aucune donnée mensuelle disponible pour la station {selected_station}.")
            else:
                # Ajouter les noms des mois
                month_names = {
                    1: 'Janvier', 2: 'Février', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin',
                    7: 'Juillet', 8: 'Août', 9: 'Septembre', 10: 'Octobre', 11: 'Novembre', 12: 'Décembre'
                }
                monthly_data['nom_mois'] = monthly_data['month'].map(month_names)
                
                # Trier par mois
                monthly_data = monthly_data.sort_values('month')
                
                # Formater les valeurs numériques
                for col in monthly_data.columns:
                    if col not in ['code_station_id', 'month', 'nom_mois'] and pd.api.types.is_numeric_dtype(monthly_data[col]):
                        monthly_data[col] = monthly_data[col].round(2)
                
                # Afficher le tableau des seuils mensuels
                st.dataframe(monthly_data[['nom_mois', 'debit_moyen', 'seuil_vigilance', 'seuil_alerte', 'seuil_alerte_renforcee', 'seuil_crise']])
                
                # Visualisation des seuils mensuels
                fig = px.line(
                    monthly_data, 
                    x='month', 
                    y=['debit_moyen', 'seuil_vigilance', 'seuil_alerte', 'seuil_alerte_renforcee', 'seuil_crise'],
                    title=f"Variation mensuelle des seuils pour la station {selected_station}",
                    labels={'month': 'Mois', 'value': 'Débit (m³/s)', 'variable': 'Type de seuil'},
                    color_discrete_map={
                        'debit_moyen': 'blue',
                        'seuil_vigilance': 'yellow',
                        'seuil_alerte': 'orange',
                        'seuil_alerte_renforcee': 'orangered',
                        'seuil_crise': 'red'
                    }
                )
                
                # Améliorer l'axe des x pour afficher les noms des mois
                fig.update_xaxes(
                    tickvals=list(range(1, 13)),
                    ticktext=list(month_names.values())
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Aucun seuil mensuel calculé.")
    
    elif display_type == "Prédictions":
        st.header(f"Prédictions de débit pour la station {selected_station}")
        
        # Obtenir le nom de la station
        station_name = station_options[station_options['code_station_id'] == selected_station]['nom_station'].iloc[0]
        
        # Filtrer les données pour la station sélectionnée
        flow_column = 'resultat_obs_elab' if 'resultat_obs_elab' in data.columns else 'resultat_obs'
        station_data = data[data['code_station_id'] == selected_station].copy()
        station_data = station_data.sort_values('date')
        
        if len(station_data) < 10:
            st.warning("Données insuffisantes pour faire des prédictions fiables. Un minimum de 10 points de données est recommandé.")
            if len(station_data) == 0:
                return
        
        # Créer et entraîner le modèle
        with st.spinner("Entraînement du modèle de prédiction..."):
            try:
                # DEBUG: Afficher les colonnes disponibles
                st.write("Colonnes disponibles dans le dataset:", data.columns.tolist())
                
                # Vérifier si saison est disponible
                if 'saison' in data.columns:
                    st.write("La colonne 'saison' est disponible et contient:", data['saison'].unique())
                
                model_info = create_prediction_model(data, selected_station, method=prediction_method.lower())
                if model_info is None:
                    st.error("Impossible de créer le modèle de prédiction. Vérifiez les données d'entrée.")
                    return
                
                # DEBUG: Afficher les informations du modèle
                st.write("Informations du modèle:", {k: v for k, v in model_info.items() if k != 'model'})
                
                forecast = make_predictions(model_info, data, selected_station, days=prediction_days)
                if forecast is None:
                    st.error("Échec de la prédiction. Vérifiez le modèle.")
                    return
            except Exception as e:
                st.error(f"Erreur lors de la création du modèle de prédiction: {str(e)}")
                st.exception(e)  # Afficher la trace complète
                return
        
        # Visualiser les prédictions
        fig = go.Figure()
        
        # Données historiques
        fig.add_trace(go.Scatter(
            x=station_data['date'],
            y=station_data[flow_column],
            mode='markers',
            name='Données historiques',
            marker=dict(color='blue', size=6)
        ))
        
        # Ajouter une ligne de tendance pour les données historiques
        fig.add_trace(go.Scatter(
            x=station_data['date'],
            y=station_data[flow_column],
            mode='lines',
            name='Tendance historique',
            line=dict(color='royalblue', width=1)
        ))
        
        # Prédictions
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['prediction'],
            mode='lines',
            name='Prédiction',
            line=dict(color='red', width=2)
        ))
        
        # Intervalle de confiance
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['date'],
            y=forecast['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='Intervalle de confiance'
        ))
        
        # Ajouter les seuils d'alerte s'ils existent
        if thresholds_df is not None:
            threshold_data = thresholds_df[thresholds_df['code_station_id'] == selected_station]
            
            if len(threshold_data) > 0:
                threshold_data = threshold_data.iloc[0]
                fig.add_hline(y=threshold_data['seuil_vigilance'], line_dash="dash", line_color="yellow", annotation_text="Vigilance")
                fig.add_hline(y=threshold_data['seuil_alerte'], line_dash="dash", line_color="orange", annotation_text="Alerte")
                fig.add_hline(y=threshold_data['seuil_crise'], line_dash="dash", line_color="red", annotation_text="Crise")
        
        # Mise en forme du graphique
        fig.update_layout(
            title=f"Prédiction des débits pour {station_name}",
            xaxis_title="Date",
            yaxis_title="Débit (m³/s)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Informations sur les dépassements de seuil prévus
        if thresholds_df is not None:
            threshold_data = thresholds_df[thresholds_df['code_station_id'] == selected_station]
            
            if len(threshold_data) > 0:
                threshold_data = threshold_data.iloc[0]
                
                try:
                    # Vérifier s'il y a des dépassements prévus
                    alert_days = forecast[forecast['prediction'] > threshold_data['seuil_alerte']]
                    crisis_days = forecast[forecast['prediction'] > threshold_data['seuil_crise']]
                    
                    st.subheader("Analyse des risques")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Jours de dépassement du seuil d'alerte prévus", 
                            len(alert_days),
                            delta=None,
                            delta_color="inverse"
                        )
                        
                        if len(alert_days) > 0:
                            st.write("Dates prévues de dépassement du seuil d'alerte:")
                            alert_dates = [d.strftime("%d/%m/%Y") for d in alert_days['date']]
                            # Limiter le nombre de dates affichées si nécessaire
                            if len(alert_dates) > 10:
                                st.write(", ".join(alert_dates[:10]) + f" et {len(alert_dates) - 10} autres jours")
                            else:
                                st.write(", ".join(alert_dates))
                    
                    with col2:
                        st.metric(
                            "Jours de dépassement du seuil de crise prévus", 
                            len(crisis_days),
                            delta=None,
                            delta_color="inverse"
                        )
                        
                        if len(crisis_days) > 0:
                            st.write("Dates prévues de dépassement du seuil de crise:")
                            crisis_dates = [d.strftime("%d/%m/%Y") for d in crisis_days['date']]
                            # Limiter le nombre de dates affichées si nécessaire
                            if len(crisis_dates) > 10:
                                st.write(", ".join(crisis_dates[:10]) + f" et {len(crisis_dates) - 10} autres jours")
                            else:
                                st.write(", ".join(crisis_dates))
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse des risques: {str(e)}")
    
    # Pied de page
    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("""
    Cette application permet de visualiser et d'analyser les données des stations hydrométriques d'Occitanie. 
    Elle calcule des seuils d'alerte basés sur les statistiques historiques et propose un modèle de prédiction des débits.
    
    **Note:** Les seuils calculés sont basés uniquement sur l'analyse statistique des données historiques et ne remplacent pas les seuils réglementaires officiels.
    """)

if __name__ == "__main__":
    main()
