# Documentation Complète : Eko - Surveillance des Rivières d'Occitanie

## Table des matières

1. [Introduction](#introduction)
2. [Installation et prérequis](#installation-et-prérequis)
3. [Structure des données](#structure-des-données)
4. [Architecture de l'application](#architecture-de-lapplication)
5. [Fonctionnalités principales](#fonctionnalités-principales)
   - [Carte des stations](#carte-des-stations)
   - [Historique des débits](#historique-des-débits)
   - [Seuils d'alerte](#seuils-dalerte)
   - [Prédictions](#prédictions)
   - [Méthodologie](#méthodologie)
6. [Modèles de prédiction](#modèles-de-prédiction)
   - [Random Forest](#random-forest)
   - [Prophet](#prophet)
7. [Calcul des seuils](#calcul-des-seuils)
8. [Interface utilisateur](#interface-utilisateur)
9. [Limites et perspectives](#limites-et-perspectives)
10. [Glossaire](#glossaire)
11. [Dépannage](#dépannage)

## Introduction

**Eko - Surveillance des Rivières d'Occitanie** est une application web dédiée à la surveillance, l'analyse et la prédiction des débits des cours d'eau dans la région Occitanie en France. Ce projet a été développé comme une preuve de concept (POC) pour démontrer comment les données hydrométriques peuvent être utilisées pour surveiller en temps réel l'état des rivières, calculer des seuils d'alerte, et prédire les évolutions futures des débits.

### Objectif du projet

L'application vise à offrir aux utilisateurs (gestionnaires de l'eau, collectivités, citoyens intéressés) un outil permettant de :

- Visualiser l'emplacement des stations de mesure hydrométrique
- Consulter l'historique des débits des rivières
- Identifier les situations potentiellement critiques grâce à des seuils d'alerte calculés statistiquement
- Prévoir l'évolution future des débits à court et moyen terme

Dans un contexte de changement climatique et de pressions croissantes sur les ressources en eau, cette application répond à un besoin de surveillance et d'anticipation pour une meilleure gestion de l'eau.

## Installation et prérequis

### Prérequis techniques

Pour exécuter l'application Eko, vous aurez besoin de :

- Python 3.7 ou supérieur
- pip (gestionnaire de paquets Python)
- Accès à internet pour l'affichage des cartes

### Dépendances principales

L'application repose sur plusieurs bibliothèques Python :

```
streamlit>=1.12.0
pandas>=1.3.0
numpy>=1.20.0
pydeck>=0.7.0
plotly>=5.3.0
prophet>=1.0.0
scikit-learn>=1.0.0
```

### Installation

1. Clonez le dépôt ou téléchargez les fichiers sources
2. Naviguez jusqu'au répertoire du projet
3. Installez les dépendances :

```bash
pip install -r requirements.txt
```

### Lancement de l'application

Pour démarrer l'application, exécutez la commande suivante depuis le répertoire du projet :

```bash
streamlit run app.py
```

L'application sera accessible via votre navigateur à l'adresse `http://localhost:8501`

## Structure des données

### Fichiers de données

L'application attend un fichier CSV principal contenant les mesures hydrométriques. Le fichier par défaut est nommé `mesures_hydro_occitanie_2020_2025_20250520_100820.csv`, mais vous pouvez utiliser vos propres fichiers via l'interface d'upload.

### Format des données attendu

Le fichier CSV doit contenir au minimum les colonnes suivantes :

| Colonne | Description | Type |
|---------|-------------|------|
| code_station_id | Identifiant unique de la station | Texte |
| nom_station | Nom de la station | Texte |
| date_obs_elab ou date_obs | Date de la mesure | Date (YYYY-MM-DD) |
| resultat_obs_elab ou resultat_obs | Valeur du débit mesuré (m³/s) | Nombre décimal |
| longitude | Coordonnée géographique (longitude) | Nombre décimal |
| latitude | Coordonnée géographique (latitude) | Nombre décimal |

Les colonnes optionnelles qui enrichissent l'analyse :

| Colonne | Description | Type |
|---------|-------------|------|
| libelle_cours_eau | Nom du cours d'eau | Texte |
| saison | Saison de la mesure (Hiver, Printemps, Été, Automne) | Texte |

### Source des données

Les données utilisées proviennent généralement de l'API Hub'Eau (Hydrométrie), qui centralise les mesures des stations de surveillance des cours d'eau français. L'application peut fonctionner avec des extractions de cette API ou toute autre source respectant le format décrit ci-dessus.

## Architecture de l'application

### Vue d'ensemble

L'application est construite avec Streamlit, un framework Python permettant de créer facilement des applications web interactives pour la data science. Elle suit une architecture modulaire où différentes fonctions sont responsables de tâches spécifiques.

### Fonctions principales

- `load_hydro_data()` : Charge et prétraite les données hydrométriques
- `extract_stations_info()` : Extrait les informations des stations
- `calculate_thresholds()` : Calcule les seuils d'alerte globaux
- `calculate_monthly_thresholds()` : Calcule les seuils d'alerte mensuels
- `create_prediction_model()` : Crée et entraîne un modèle de prédiction
- `make_predictions()` : Génère des prédictions avec le modèle
- `evaluate_model()` : Évalue la performance du modèle
- `display_station_map()` : Affiche la carte des stations
- `main()` : Fonction principale qui orchestre l'application

### Flux de données

1. Chargement des données brutes (CSV)
2. Prétraitement et nettoyage
3. Calcul des statistiques et seuils
4. Entraînement des modèles de prédiction (si nécessaire)
5. Affichage via l'interface utilisateur Streamlit

## Fonctionnalités principales

### Carte des stations

La carte interactive des stations hydrométriques affiche l'emplacement de toutes les stations de mesure sur une carte de la région Occitanie. Chaque station est représentée par un point dont la couleur indique son état actuel :

- **Bleu** : Station avec débit normal
- **Orange** : Station au-dessus du seuil d'alerte
- **Rouge** : Station au-dessus du seuil de crise
- **Gris** : Station sans données de débit récentes

#### Fonctionnalités de la carte
- Zoom avant/arrière
- Survol des points pour afficher des informations détaillées
- Tableau récapitulatif des stations en dessous de la carte

Cette visualisation permet d'identifier rapidement les zones géographiques où les rivières présentent des débits anormaux.

### Historique des débits

Cette section permet de consulter l'historique des mesures de débit pour une station spécifique. Les données sont présentées sous forme de graphique temporel, avec les seuils d'alerte affichés comme des lignes de référence.

#### Éléments affichés
- Graphique des débits au fil du temps
- Statistiques clés (débit minimum, moyen, maximum, écart-type)
- Répartition saisonnière des débits (si les données de saison sont disponibles)

Cette visualisation permet d'identifier les tendances, les variations saisonnières et les événements exceptionnels dans l'historique d'une station.

### Seuils d'alerte

Cette section présente les différents seuils d'alerte calculés à partir des données historiques. Deux types de seuils sont disponibles :

#### Seuils globaux
Calculés sur l'ensemble de la période disponible pour chaque station :
- Seuil de vigilance (75ème percentile)
- Seuil d'alerte (90ème percentile)
- Seuil d'alerte renforcée (95ème percentile)
- Seuil de crise (98ème percentile)

#### Seuils mensuels
Calculés pour chaque mois de l'année, permettant de tenir compte des variations saisonnières normales.

Les seuils sont présentés sous forme de tableaux et de graphiques, facilitant la compréhension des niveaux critiques pour chaque station et période de l'année.

### Prédictions

Cette section utilise des modèles d'apprentissage automatique pour prédire l'évolution future des débits. Elle comprend deux onglets :

#### Prédictions futures
- Graphique montrant l'historique récent et les prédictions futures
- Intervalle de confiance autour des prédictions
- Analyse des risques de dépassement des seuils d'alerte

#### Évaluation du modèle
- Métriques de performance du modèle (MAE, RMSE, R², MAPE)
- Comparaison graphique des prédictions et valeurs réelles sur un jeu de test
- Interprétation des résultats

L'utilisateur peut choisir entre deux méthodes de prédiction (Random Forest ou Prophet) et définir l'horizon de prédiction (nombre de jours).

### Méthodologie

Cette section explicative détaille les aspects techniques de l'application à travers quatre onglets :

#### Sources de données
- Origine des données (API Hub'Eau)
- Processus de collecte
- Volume et répartition des données

#### Calcul des seuils
- Méthodologie statistique
- Signification des différents seuils
- Approche des seuils mensuels

#### Modèles de prédiction
- Description des méthodes utilisées
- Avantages et limites de chaque modèle
- Processus d'évaluation

#### Limites et perspectives
- Limites actuelles de l'application
- Pistes d'amélioration futures
- Références aux méthodes officielles

## Modèles de prédiction

L'application propose deux approches différentes pour la prédiction des débits futurs :

### Random Forest

Random Forest est un algorithme d'apprentissage supervisé basé sur un ensemble d'arbres de décision.

#### Caractéristiques
- **Variables utilisées** : mois, jour de l'année, saison (si disponible)
- **Hyperparamètres** : 100 arbres, profondeur maximale automatique
- **Validation** : Séparation 80% entraînement / 20% test (validation temporelle)

#### Avantages
- Capture les relations non-linéaires complexes
- Robuste face aux valeurs aberrantes
- Performant même avec des jeux de données de taille modeste

#### Limites
- Peut avoir tendance à surajuster les données (overfitting)
- Moins adapté pour capturer les tendances à très long terme
- Intervalles de confiance approximatifs

### Prophet

Prophet est un modèle de série temporelle développé par Facebook, spécialement conçu pour les prévisions avec forte saisonnalité.

#### Caractéristiques
- Décompose la série en tendance, saisonnalité et composante résiduelle
- Intègre des facteurs de saisonnalité quotidienne
- Fournit des intervalles de prédiction natifs

#### Avantages
- Capture efficacement les tendances et cycles saisonniers
- Gère bien les données manquantes ou irrégulières
- Intervalles de confiance intégrés et robustes

#### Limites
- Moins adaptable aux relations complexes non-saisonnières
- Performances parfois inférieures à Random Forest sur des séries avec peu de saisonnalité
- Temps de calcul plus important

## Calcul des seuils

### Méthodologie

Les seuils d'alerte sont calculés à partir des statistiques historiques des débits pour chaque station. L'application utilise une approche basée sur les percentiles de la distribution des débits observés.

#### Seuils calculés

1. **Seuil de vigilance** (75ème percentile)
   - Représente le niveau au-delà duquel une attention particulière est recommandée
   - 75% des observations historiques sont inférieures à ce seuil

2. **Seuil d'alerte** (90ème percentile)
   - Indique un niveau élevé nécessitant une surveillance accrue
   - 90% des observations historiques sont inférieures à ce seuil

3. **Seuil d'alerte renforcée** (95ème percentile)
   - Signale un niveau très élevé pouvant conduire à des restrictions
   - 95% des observations historiques sont inférieures à ce seuil

4. **Seuil de crise** (98ème percentile)
   - Indique un niveau critique requérant des mesures d'urgence
   - 98% des observations historiques sont inférieures à ce seuil

### Seuils mensuels

L'application calcule également des seuils spécifiques pour chaque mois de l'année, permettant de prendre en compte les variations saisonnières normales des débits. Cette approche permet une plus grande précision dans la détection des anomalies, car un débit considéré comme élevé en été peut être normal en hiver.

La méthode de calcul est identique à celle des seuils globaux, mais appliquée séparément pour chaque mois de l'année.

### Note importante

Ces seuils sont calculés à titre indicatif à partir des données historiques et ne remplacent pas les seuils réglementaires officiels définis par les autorités compétentes (Services de Prévision des Crues, DREAL, etc.).

## Interface utilisateur

### Organisation générale

L'interface de l'application est organisée en deux zones principales :
- **Sidebar (barre latérale gauche)** : Contrôles et filtres
- **Zone principale** : Visualisations et informations détaillées

### Contrôles dans la sidebar

1. **Informations sur les données**
   - Nombre de stations
   - Nombre total d'observations
   - Période couverte

2. **Sélection de la station**
   - Liste déroulante avec identifiant et nom de station

3. **Type d'affichage**
   - Carte des stations
   - Historique des débits
   - Seuils d'alerte
   - Prédictions
   - Méthodologie

4. **Options spécifiques**
   - Pour les prédictions : nombre de jours, méthode

### Navigation et utilisation

1. Commencez par sélectionner une station dans la liste déroulante
2. Choisissez le type d'affichage souhaité
3. Explorez les visualisations et données présentées
4. Ajustez les paramètres spécifiques si nécessaire

L'interface est responsive et s'adapte à différentes tailles d'écran, bien qu'elle soit optimisée pour une utilisation sur ordinateur.

## Limites et perspectives

### Limites actuelles

#### Limites des données
- Certaines stations disposent de peu d'observations
- Données parfois discontinues dans le temps
- Absence de certaines variables contextuelles (précipitations, température)

#### Limites des modèles
- Prédictions à plus de 30 jours peuvent manquer de fiabilité
- Difficulté à prévoir des événements extrêmes
- Modèles non calibrés pour les crues exceptionnelles

#### Limites techniques
- Temps de chargement initial parfois long
- Absence de système d'alerte automatisé
- Interface mobile limitée

### Perspectives d'amélioration

#### Enrichissement des données
- Intégration de données météorologiques (précipitations, températures)
- Ajout de caractéristiques géographiques des bassins versants
- Collecte de données historiques plus complètes

#### Améliorations techniques
- Développement de modèles hybrides combinant différentes approches
- Utilisation de techniques d'apprentissage profond pour les séries temporelles
- Intégration d'algorithmes de détection d'anomalies plus sophistiqués

#### Évolutions fonctionnelles
- Système d'alerte par email ou SMS
- Interface mobile optimisée
- Intégration avec d'autres sources de données environnementales
- Module de comparaison entre stations

## Glossaire

**Débit** : Volume d'eau s'écoulant à travers une section de rivière par unité de temps, généralement exprimé en mètres cubes par seconde (m³/s).

**Station hydrométrique** : Installation fixe équipée d'instruments de mesure permettant de suivre en continu les hauteurs d'eau et/ou les débits d'un cours d'eau.

**Percentile** : Valeur en dessous de laquelle se trouve un certain pourcentage des observations. Par exemple, le 90ème percentile est la valeur en dessous de laquelle se trouvent 90% des observations.

**MAE (Mean Absolute Error)** : Erreur absolue moyenne, mesure la moyenne des écarts absolus entre les prédictions et les valeurs réelles.

**RMSE (Root Mean Square Error)** : Racine carrée de l'erreur quadratique moyenne, mesure similaire au MAE mais pénalisant davantage les erreurs importantes.

**R²** : Coefficient de détermination, mesure la proportion de la variance expliquée par le modèle (entre 0 et 1).

**MAPE (Mean Absolute Percentage Error)** : Pourcentage d'erreur absolue moyen, exprime l'erreur en pourcentage par rapport aux valeurs réelles.

**Random Forest** : Algorithme d'apprentissage automatique basé sur un ensemble d'arbres de décision.

**Prophet** : Modèle de prévision de séries temporelles développé par Facebook, particulièrement adapté aux données avec des saisonnalités multiples.

**Overfitting** : Surajustement, situation où un modèle s'adapte trop précisément aux données d'entraînement et perd en capacité de généralisation.

## Dépannage

### Problèmes courants et solutions

#### L'application ne se lance pas
- Vérifiez que toutes les dépendances sont installées : `pip install -r requirements.txt`
- Assurez-vous d'avoir Python 3.7 ou supérieur : `python --version`
- Vérifiez les logs d'erreur dans la console

#### Le fichier de données n'est pas reconnu
- Vérifiez que le format du fichier CSV est correct
- Assurez-vous que les colonnes requises sont présentes
- Utilisez l'option d'upload de fichier dans l'interface

#### Les prédictions ne s'affichent pas
- Vérifiez que la station sélectionnée dispose de suffisamment de données (minimum 10 points)
- Essayez de réduire l'horizon de prédiction
- Testez l'autre méthode de prédiction

#### La carte ne s'affiche pas correctement
- Vérifiez votre connexion internet
- Assurez-vous que les coordonnées géographiques sont dans un format valide
- Actualisez la page

---

*Cette documentation a été rédigée pour le projet POC "Eko - Surveillance des Rivières d'Occitanie", version 1.0, mai 2025.*