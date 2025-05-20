import requests
import pandas as pd
import random
from datetime import datetime, timedelta
from tqdm import tqdm

# Codes départementaux de l'Occitanie
departements_occitanie = ['09', '11', '12', '30', '31', '32', '34', '46', '48', '65', '66', '81', '82']

def get_stations_by_region(departements, limit_per_dept=5):  # Augmenté le nombre de stations par défaut
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

def check_station_has_2025_data(station_code):
    """Vérifie si la station a des données en 2025"""
    print(f"  Vérification des données 2025 pour la station {station_code}...")
    
    # URL pour les données élaborées (plus probables d'avoir des données historiques)
    url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
    
    # Période de test pour 2025
    start_date = "2025-01-01"
    end_date = "2025-05-20"  # Date actuelle (aujourd'hui)
    
    # Essayer d'abord les débits (Q) puis les hauteurs (H)
    for data_type in ["QmnJ", "HmnJ"]:
        params = {
            "code_entite": station_code,
            "grandeur_hydro_elab": data_type,
            "date_debut_obs_elab": start_date,
            "date_fin_obs_elab": end_date,
            "size": 1  # Une seule observation suffit pour vérifier
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if "data" in data and len(data["data"]) > 0:
                print(f"  ✓ Station {station_code} a des données {data_type} en 2025")
                return True
                
        except Exception as e:
            print(f"  ! Erreur lors de la vérification pour {station_code}: {str(e)}")
    
    # Essayer également les données temps réel
    url_tr = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr"
    
    for data_type in ["Q", "H"]:
        params = {
            "code_entite": station_code,
            "grandeur_hydro": data_type,
            "date_debut_obs": start_date,
            "date_fin_obs": end_date,
            "size": 1
        }
        
        try:
            response = requests.get(url_tr, params=params)
            data = response.json()
            
            if "data" in data and len(data["data"]) > 0:
                print(f"  ✓ Station {station_code} a des données {data_type} temps réel en 2025")
                return True
                
        except Exception:
            pass
    
    print(f"  ✗ Station {station_code} n'a pas de données en 2025")
    return False

def get_station_data_for_period(station_code, station_name, start_year, end_year, samples_per_year=50):
    """Récupère plus de données pour une station sur plusieurs années"""
    all_observations = []
    
    # Types de données à essayer, par ordre de priorité
    data_types = ["QmnJ", "HmnJ", "Q", "H"]
    
    # Pour chaque année dans la période
    for year in range(start_year, end_year + 1):
        print(f"  Récupération des données pour {station_name} en {year}...")
        year_data_found = False
        
        # Essayer différents types de données
        for data_type in data_types:
            if year_data_found:
                break  # Si on a trouvé des données pour cette année, passer à la suivante
                
            # Déterminer les paramètres API selon le type de données
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
            
            # Période pour l'année en cours
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
                response = requests.get(url, params=params)
                data = response.json()
                
                if "data" in data and len(data["data"]) > 0:
                    observations = data["data"]
                    print(f"  ✓ {len(observations)} observations {data_type} trouvées pour {year}")
                    
                    # Échantillonner pour avoir des données bien réparties sur l'année
                    if len(observations) > samples_per_year:
                        # Stratifier l'échantillonnage pour avoir des données réparties dans l'année
                        interval = max(1, len(observations) // samples_per_year)
                        indices = list(range(0, len(observations), interval))[:samples_per_year]
                        sampled_observations = [observations[i] for i in indices]
                    else:
                        sampled_observations = observations
                    
                    # Ajouter les métadonnées
                    for obs in sampled_observations:
                        obs['code_station_id'] = station_code
                        obs['nom_station'] = station_name
                        obs['type_donnee'] = data_type
                        
                        # Déterminer la saison
                        date_field = 'date_obs_elab' if 'date_obs_elab' in obs else 'date_obs'
                        if date_field in obs:
                            date_str = obs[date_field]
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
                    
                    all_observations.extend(sampled_observations)
                    year_data_found = True
                    
            except Exception as e:
                print(f"  ! Erreur pour {data_type} en {year}: {str(e)}")
    
    return all_observations

def main():
    # 1. Récupérer des stations d'Occitanie
    stations_df = get_stations_by_region(departements_occitanie, limit_per_dept=4)  # 4 stations par département
    
    if len(stations_df) == 0:
        print("Aucune station trouvée, impossible de continuer.")
        return
    
    # 2. Filtrer les stations qui ont des données en 2025
    print("\nVérification des stations avec données en 2025...")
    stations_with_2025_data = []
    
    # Tester au maximum 20 stations pour ne pas surcharger l'API
    test_stations = stations_df.sample(min(20, len(stations_df)))
    
    for _, station in test_stations.iterrows():
        code_station = station['code_station']
        if check_station_has_2025_data(code_station):
            stations_with_2025_data.append(station)
    
    if not stations_with_2025_data:
        print("Aucune station avec des données en 2025 n'a été trouvée. Essayez d'augmenter le nombre de stations testées.")
        return
    
    print(f"\n{len(stations_with_2025_data)} stations avec des données en 2025 identifiées.")
    
    # 3. Collecter des données historiques pour chaque station sur plusieurs années
    all_observations = []
    
    # Période souhaitée: plusieurs années précédant 2025
    start_year = 2020  # Commencer en 2020
    end_year = 2025    # Jusqu'à 2025
    
    print(f"\nCollecte des données de {start_year} à {end_year} pour les stations sélectionnées...\n")
    
    # Pour chaque station validée, récupérer les données
    for i, station in enumerate(stations_with_2025_data):
        code_station = station['code_station']
        nom_station = station['libelle_station']
        
        print(f"Station {i+1}/{len(stations_with_2025_data)}: {nom_station} ({code_station})")
        
        # Récupérer plus de données sur toute la période
        observations = get_station_data_for_period(code_station, nom_station, start_year, end_year, samples_per_year=100)
        
        if observations and len(observations) > 0:
            all_observations.extend(observations)
            print(f"  ✓ {len(observations)} observations collectées pour {nom_station}\n")
        else:
            print(f"  ✗ Aucune donnée récupérée pour {nom_station}\n")
    
    # 4. Génération du CSV final
    if all_observations:
        # Convertir en DataFrame
        observations_df = pd.DataFrame(all_observations)
        
        # Créer un nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"mesures_hydro_occitanie_2020_2025_{timestamp}.csv"
        
        # Sauvegarder dans un CSV
        observations_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\nTous les résultats ont été sauvegardés dans {output_csv}")
        print(f"Nombre total d'observations: {len(observations_df)}")
        print(f"Nombre de stations: {observations_df['code_station_id'].nunique()}")
        
        # Afficher la plage temporelle
        date_cols = [col for col in observations_df.columns if 'date' in col.lower() and 'prod' not in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            try:
                min_date = min(observations_df[date_col].apply(lambda x: x.split('T')[0] if 'T' in str(x) else x))
                max_date = max(observations_df[date_col].apply(lambda x: x.split('T')[0] if 'T' in str(x) else x))
                print(f"Période couverte: du {min_date} au {max_date}")
            except:
                pass
        
        # Afficher les statistiques
        if 'type_donnee' in observations_df.columns:
            print("\nRépartition par type de données:")
            print(observations_df['type_donnee'].value_counts())
        
        if 'saison' in observations_df.columns:
            print("\nRépartition par saison:")
            print(observations_df['saison'].value_counts())
            
        if 'year' not in observations_df.columns and date_cols:
            try:
                observations_df['year'] = observations_df[date_col].apply(
                    lambda x: datetime.strptime(x.split('T')[0], "%Y-%m-%d").year if 'T' in str(x) else x[:4])
                print("\nRépartition par année:")
                print(observations_df['year'].value_counts())
            except:
                pass
    else:
        print("\nAucune donnée n'a pu être récupérée.")

if __name__ == "__main__":
    main()