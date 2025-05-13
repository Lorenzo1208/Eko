import requests
import pandas as pd
import random
from datetime import datetime, timedelta
from tqdm import tqdm

# Codes départementaux de l'Occitanie
departements_occitanie = ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82']

def get_stations_by_region(departements, limit_per_dept=3):
    """Récupère les stations hydrométriques des départements indiqués"""
    print("Récupération des stations hydrométriques en Occitanie...")
    
    # Liste pour stocker toutes les stations
    all_stations = []
    
    # URL de l'API pour le référentiel des stations
    url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations"
    
    # Pour chaque département, récupérer ses stations
    for dept in tqdm(departements):
        # Paramètres de la requête
        params = {
            "code_departement": dept,
            "size": 1000,  # Nombre maximum de résultats par page
            "format": "json"
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Si des stations sont trouvées, en ajouter un nombre limité à notre liste
            if "data" in data and len(data["data"]) > 0:
                # Filtrer seulement les stations en service pour de meilleures chances d'avoir des données
                active_stations = [s for s in data["data"] if s.get("en_service") == True]
                if active_stations:
                    # Prendre un échantillon aléatoire limité
                    sample_size = min(limit_per_dept, len(active_stations))
                    selected_stations = random.sample(active_stations, sample_size)
                    all_stations.extend(selected_stations)
                    print(f"  - {dept} : {len(selected_stations)} stations ajoutées (sur {len(active_stations)} actives)")
                else:
                    # Si pas de stations actives, essayer quand même avec des stations inactives
                    sample_size = min(limit_per_dept, len(data["data"]))
                    selected_stations = random.sample(data["data"], sample_size)
                    all_stations.extend(selected_stations)
                    print(f"  - {dept} : {len(selected_stations)} stations ajoutées (toutes inactives)")
            else:
                print(f"  - {dept} : Aucune station trouvée")
                
        except Exception as e:
            print(f"Erreur lors de la récupération des stations du département {dept}: {str(e)}")
    
    # Créer un DataFrame avec les stations
    df_stations = pd.DataFrame(all_stations)
    
    return df_stations

def get_station_data_window(station_code, station_name, start_year, end_year, data_type="QmnJ"):
    """Récupère les données d'une station sur une période plus large puis échantillonne"""
    # Déterminer les paramètres de l'API selon le type de données
    if data_type in ["QmnJ", "HmnJ"]:
        # Pour les données journalières
        url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
        date_debut_param = "date_debut_obs_elab"
        date_fin_param = "date_fin_obs_elab"
        grandeur_param = "grandeur_hydro_elab"
    else:
        # Pour les données en temps réel (moins susceptibles d'avoir un historique)
        url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr"
        date_debut_param = "date_debut_obs"
        date_fin_param = "date_fin_obs"
        grandeur_param = "grandeur_hydro"
    
    # Essayer d'abord une année entière
    for year in range(start_year, end_year + 1):
        # Créer une fenêtre temporelle d'un an
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        # Paramètres de la requête
        params = {
            "code_entite": station_code,
            grandeur_param: data_type,
            date_debut_param: start_date,
            date_fin_param: end_date,
            "size": 10000,  # Maximum de résultats
            "sort": "asc"
        }
        
        try:
            print(f"  Récupération des données {data_type} pour {station_name} en {year}...")
            response = requests.get(url, params=params)
            data = response.json()
            
            if "data" in data and len(data["data"]) > 0:
                observations = data["data"]
                print(f"  ✓ {len(observations)} observations trouvées pour {year}")
                
                # Échantillonner les données pour avoir des mesures variées
                if len(observations) > 20:
                    # Si beaucoup de données, prendre un échantillon aléatoire
                    sample_size = min(20, len(observations))
                    sampled_observations = random.sample(observations, sample_size)
                    print(f"  → Échantillonnage de {sample_size} observations variées")
                else:
                    sampled_observations = observations
                
                # Ajouter des métadonnées à chaque observation
                for obs in sampled_observations:
                    obs['code_station_id'] = station_code
                    obs['nom_station'] = station_name
                    obs['type_donnee'] = data_type
                    
                    # Déterminer la saison
                    if 'date_obs_elab' in obs:
                        date_str = obs['date_obs_elab']
                    elif 'date_obs' in obs:
                        date_str = obs['date_obs']
                    else:
                        continue
                        
                    try:
                        date_obj = datetime.strptime(date_str.split('T')[0], "%Y-%m-%d")
                        month = date_obj.month
                        
                        if 3 <= month <= 5:
                            obs['saison'] = "Printemps"
                        elif 6 <= month <= 8:
                            obs['saison'] = "Été"
                        elif 9 <= month <= 11:
                            obs['saison'] = "Automne"
                        else:
                            obs['saison'] = "Hiver"
                    except:
                        pass
                
                return sampled_observations
            
        except Exception as e:
            print(f"  ! Erreur pour {station_name} en {year}: {str(e)}")
    
    print(f"  ✗ Aucune donnée trouvée pour {station_name} entre {start_year} et {end_year}")
    return None

def try_multiple_data_types(station_code, station_name, start_year, end_year):
    """Essaie plusieurs types de données pour s'assurer d'obtenir des résultats"""
    # Ordre de priorité pour les types de données (du plus probable au moins probable)
    data_types = ["QmnJ", "HmnJ", "Q", "H"]
    
    for data_type in data_types:
        observations = get_station_data_window(station_code, station_name, start_year, end_year, data_type)
        if observations and len(observations) > 0:
            return observations
    
    # Si aucun type de données ne fonctionne, essayer avec des années plus récentes
    for data_type in ["QmnJ", "HmnJ"]:  # Se concentrer sur les données journalières pour les années récentes
        for year in range(2020, 2024):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            if data_type in ["QmnJ", "HmnJ"]:
                # Pour les données journalières
                url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
                date_debut_param = "date_debut_obs_elab"
                date_fin_param = "date_fin_obs_elab"
                grandeur_param = "grandeur_hydro_elab"
            else:
                # Pour les données en temps réel
                url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr"
                date_debut_param = "date_debut_obs"
                date_fin_param = "date_fin_obs"
                grandeur_param = "grandeur_hydro"
            
            # Paramètres de la requête
            params = {
                "code_entite": station_code,
                grandeur_param: data_type,
                date_debut_param: start_date,
                date_fin_param: end_date,
                "size": 10000,
                "sort": "asc"
            }
            
            try:
                print(f"  Tentative avec {data_type} pour {year}...")
                response = requests.get(url, params=params)
                data = response.json()
                
                if "data" in data and len(data["data"]) > 0:
                    observations = data["data"]
                    
                    # Échantillonner pour avoir des données variées
                    if len(observations) > 15:
                        sampled_observations = random.sample(observations, 15)
                    else:
                        sampled_observations = observations
                    
                    # Ajouter les métadonnées
                    for obs in sampled_observations:
                        obs['code_station_id'] = station_code
                        obs['nom_station'] = station_name
                        obs['type_donnee'] = data_type
                        
                        # Déterminer la saison
                        if 'date_obs_elab' in obs:
                            date_str = obs['date_obs_elab']
                        elif 'date_obs' in obs:
                            date_str = obs['date_obs']
                        else:
                            continue
                            
                        try:
                            date_obj = datetime.strptime(date_str.split('T')[0], "%Y-%m-%d")
                            month = date_obj.month
                            
                            if 3 <= month <= 5:
                                obs['saison'] = "Printemps"
                            elif 6 <= month <= 8:
                                obs['saison'] = "Été"
                            elif 9 <= month <= 11:
                                obs['saison'] = "Automne"
                            else:
                                obs['saison'] = "Hiver"
                        except:
                            pass
                    
                    print(f"  ✓ {len(sampled_observations)} observations trouvées pour {year}")
                    return sampled_observations
                    
            except Exception as e:
                print(f"  ! Erreur: {str(e)}")
    
    return None

def main():
    # 1. Récupérer un échantillon de stations d'Occitanie
    stations_df = get_stations_by_region(departements_occitanie, limit_per_dept=2)
    
    if len(stations_df) == 0:
        print("Aucune station trouvée, impossible de continuer.")
        return
    
    # Sauvegarder la liste des stations dans un CSV
    stations_csv = "stations_occitanie_selection.csv"
    stations_df.to_csv(stations_csv, index=False, encoding='utf-8')
    print(f"\nListe des stations sauvegardée dans {stations_csv}")
    print(f"Nombre de stations récupérées: {len(stations_df)}")
    
    # 2. Collecter des données pour chaque station
    all_observations = []
    
    # Nombre de stations à traiter (limiter pour ne pas surcharger l'API)
    max_stations = min(6, len(stations_df))
    selected_stations = stations_df.sample(max_stations)
    
    print(f"\nCollecte de données variées pour {max_stations} stations...\n")
    
    # Définir des fenêtres temporelles différentes pour chaque station
    time_windows = [
        (2018, 2018),  # 2018 (crues)
        (2019, 2019),  # 2019
        (2020, 2020),  # 2020 (COVID)
        (2021, 2021),  # 2021
        (2022, 2022),  # 2022 (sécheresse)
        (2023, 2023)   # 2023
    ]
    
    station_count = 0
    
    for i, (_, station) in enumerate(selected_stations.iterrows()):
        code_station = station['code_station']
        nom_station = station['libelle_station']
        
        print(f"Station {i+1}/{max_stations}: {nom_station} ({code_station})")
        
        # Choisir une fenêtre temporelle pour cette station
        window = random.choice(time_windows)
        start_year, end_year = window
        
        # Essayer d'obtenir des données pour cette station
        observations = try_multiple_data_types(code_station, nom_station, start_year, end_year)
        
        if observations and len(observations) > 0:
            all_observations.extend(observations)
            station_count += 1
            print(f"  ✓ Données collectées avec succès pour {nom_station}\n")
        else:
            print(f"  ✗ Impossible de trouver des données pour {nom_station}\n")
    
    # 3. Génération du CSV final
    if all_observations:
        # Convertir en DataFrame
        observations_df = pd.DataFrame(all_observations)
        
        # Mélanger les observations pour plus de variété
        observations_df = observations_df.sample(frac=1).reset_index(drop=True)
        
        # Créer un nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"mesures_hydro_variees_{timestamp}.csv"
        
        # Sauvegarder dans un CSV
        observations_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\nTous les résultats ont été sauvegardés dans {output_csv}")
        print(f"Nombre total d'observations: {len(observations_df)}")
        print(f"Nombre de stations avec données: {observations_df['code_station_id'].nunique()}")
        
        # Si dates disponibles, afficher la plage temporelle
        date_cols = [col for col in observations_df.columns if 'date' in col.lower() and 'prod' not in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                min_date = min(observations_df[date_col].apply(lambda x: x.split('T')[0] if 'T' in str(x) else x))
                max_date = max(observations_df[date_col].apply(lambda x: x.split('T')[0] if 'T' in str(x) else x))
                print(f"Période couverte: du {min_date} au {max_date}")
            except:
                pass
        
        # Afficher les statistiques par type de données et saison si disponible
        if 'type_donnee' in observations_df.columns:
            print("\nRépartition par type de données:")
            print(observations_df['type_donnee'].value_counts())
        
        if 'saison' in observations_df.columns:
            print("\nRépartition par saison:")
            print(observations_df['saison'].value_counts())
    else:
        print("\nAucune donnée n'a pu être récupérée.")

if __name__ == "__main__":
    main()