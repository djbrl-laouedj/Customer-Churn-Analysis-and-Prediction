# Prédiction du churn client

## 🔍 Présentation du projet

**Le churn client** est un enjeu majeur pour de nombreuses entreprises, en particulier dans le secteur bancaire.
Ce projet vise à prédire si un client est susceptible de **quitter l’entreprise** à l’aide de modèles de machine learning supervisés.

Le projet se concentre sur :

- La gestion des problèmes de classification déséquilibrée

- La comparaison de plusieurs modèles de machine learning

- L’optimisation des hyperparamètres

- L’évaluation des modèles à l’aide de métriques adaptées telles que le ROC-AUC, le F1-score et le Recall

## Modèles utilisés

Les **modèles** suivants ont été implémentés et comparés :

- Régression logistique

- Arbre de décision

- Forêt aléatoire (Random Forest)

- Bagging Classifier

- K plus proches voisins (KNN)

- XGBoost

Enfin, une optimisation des hyperparamètres a été réalisée à l’aide de **RandomizedSearchCV** sur les modèles Random Forest et XGBoost.

## Stratégie d’évaluation

Étant donné que le churn client est un problème de classification fortement déséquilibré, **la précision (accuracy) seule n’est pas suffisante**.

La métrique principale utilisée est :

- ROC-AUC, qui mesure la capacité du modèle à classer correctement les clients du plus à risque au moins à risque.

Métriques complémentaires :

- F1-score

- Recall

- Matrice de confusion

- Courbe ROC

## Structure du projet
```
Customer-Churn-Analysis-and-Prediction/

│── my_streamlit_app_vf.py   # Interface Streamlit

│── CustomerChurn_ML.ipynb   # Notebook du projet

│── requirements.txt        # Versions exactes des dépendances testées

│── README.md               # Documentation (anglais)

│── README_FR.md            # Documentation (français)

│── .gitignore              # Fichiers ignorés

└── Caixa Banco.csv          # Données clients bancaires
```

## Installation

1. Cloner le dépôt
```
git clone https://github.com/djbrl-laouedj/Customer-Churn-Analysis-and-Prediction.git
```
```
cd Customer-Churn-Analysis-and-Prediction
```

2. Créer un environnement virtuel (recommandé)
```
python -m venv venv
```
```
source venv/bin/activate   # Sous Windows : venv\Scripts\activate
```

3. Installer les dépendances
```
pip install -r requirements.txt
```

## Utilisation du projet

Exécution du notebook

Exécuter les cellules **à la suite** afin de :

- Charger et prétraiter les données

- Entraîner plusieurs modèles

- Effectuer l’optimisation des hyperparamètres

- Évaluer et comparer les performances des modèles

- Interface Streamlit

## Si vous souhaitez lancer la démo Streamlit :

**Sur Google Colab :**

Créer un compte : https://ngrok.com

Récupérer le token d’authentification : https://dashboard.ngrok.com/get-started/your-authtoken

Ajouter le code suivant à la fin du script / code :
```
from pyngrok import ngrok
ngrok.set_auth_token("<YOUR_NGROK_TOKEN>")
```

Lancer Streamlit :
```
!streamlit run my_streamlit_app_vf.py &>/dev/null &
```

Exposer l’application :
```
public_url = ngrok.connect(8501)
public_url
```

Redémarrer ngrok proprement si nécessaire :
```
from pyngrok import ngrok
try:
    ngrok.kill()
except:
    pass
```

**Sur Visual Studio Code :**
```
streamlit run my_streamlit_app_vf.py
```

## Guide utilisateur

L’application est organisée en deux pages principales, accessibles via le menu de navigation à gauche.

<img width="398" height="250" alt="image" src="https://github.com/user-attachments/assets/538f4faa-b250-4f22-94fe-d21930a38f3d" />

### Page 1 - Prédiction du churn client

Cette page permet de prédire le risque de churn d’un client individuel et de comprendre les raisons de cette prédiction.

**1. Sélection du modèle**

<img width="358" height="302" alt="image" src="https://github.com/user-attachments/assets/cee391bf-6821-44ac-9915-281bb1fbe9d3" />

**Choisissez un modèle de machine learning dans la liste déroulante :**

(Les 3 modèles les plus performants)

- XGBoost

- Random Forest

- Bagging

Le modèle sélectionné est utilisé pour calculer la **probabilité de churn**.

**2. Seuil de décision**

<img width="285" height="104" alt="image" src="https://github.com/user-attachments/assets/95c3528c-4ac4-460d-a75e-e3a00247a319" />

**Ajustez le seuil de décision à l’aide du curseur.**

Ce seuil représente un paramètre métier permettant de décider si un client est considéré à risque.

Un seuil plus élevé rend la décision plus conservatrice.

**3. Profil client**

<img width="285" height="421" alt="image" src="https://github.com/user-attachments/assets/1e09f4ff-7517-4df5-b963-77ab34ca6358" />

<img width="272" height="469" alt="image" src="https://github.com/user-attachments/assets/0fc3a0eb-57a6-414c-844f-6c228b1b8b38" />

**L’utilisateur peut simuler un client en ajustant les paramètres suivants :**

- Score de crédit

- Âge

- Ancienneté

- Solde du compte

- Nombre de produits

- Salaire estimé

- Pays

- Genre

- Possession d’une carte de crédit

- Statut de client actif

Ces variables constituent **le profil client** utilisé par le modèle.

**4. Lancer la prédiction**

<img width="923" height="244" alt="image" src="https://github.com/user-attachments/assets/025dc7a2-fb1b-42ba-80bd-1bc8a72c1f93" />

Cliquez sur le bouton **« Analyser le risque »** pour lancer l’analyse.

L’application affiche :

- La probabilité de churn prédite

- Le seuil de décision

- La décision finale (Risque faible / Risque élevé), basée sur la comparaison probabilité / seuil

### Explication de la prédiction (explicabilité locale)

Après la prédiction, l’application explique pourquoi le modèle a produit ce résultat.

**Impact des variables (local)**

<img width="490" height="329" alt="image" src="https://github.com/user-attachments/assets/b5fa2921-be8f-481b-afae-aac0a12da5ab" />

**Une liste des variables les plus influentes est affichée.**

Chaque variable indique :

- Le sens de son impact (augmentation ou diminution du risque)

- Sa contribution relative à la prédiction finale

Cela permet de comprendre quelles caractéristiques du client influencent **le risque de churn**.

### Explicabilité du modèle (globale – SHAP)

Au-delà des prédictions individuelles, l’application fournit une explicabilité globale :

<img width="889" height="416" alt="image" src="https://github.com/user-attachments/assets/c17ad917-3dc6-485d-96e9-8acc2a915901" />

**Visualisation de l’importance des variables basée sur SHAP**

Identification des variables les plus influentes sur l’ensemble du jeu de données

Mise en évidence des facteurs structurels du churn :

- Activité du client

- Âge

- Nombre de produits

- Facteurs géographiques

Cette section est utile pour l’interprétation du modèle et l’aide à la décision stratégique.

### Page 2 - Suivi des données & EDA

Cette page offre une vue analytique globale du churn client.

<img width="844" height="235" alt="image" src="https://github.com/user-attachments/assets/0844c7ce-c803-4b58-97c1-8e13efa86c99" />

**Indicateurs clés (KPIs)**

- Taux de churn global

- Nombre total de clients analysés

- Pays présentant le churn le plus élevé

- Segment client le plus critique

### Analyse exploratoire des données (EDA)

La page inclut plusieurs visualisations :

**Répartition globale du churn :**

<img width="921" height="414" alt="image" src="https://github.com/user-attachments/assets/76437006-137a-4ff9-b423-b9930893421d" />

**Taux de churn par genre :**

<img width="858" height="399" alt="image" src="https://github.com/user-attachments/assets/d87d3904-5948-43ac-8894-b17887ee1ba3" />

**Taux de churn par pays :**

<img width="858" height="387" alt="image" src="https://github.com/user-attachments/assets/a20131c5-33e2-4958-95e8-ed23b1584ada" />

**Heatmap du churn par segment (âge × nombre de produits) :**

<img width="886" height="408" alt="image" src="https://github.com/user-attachments/assets/570d1945-2afd-44fd-ac06-b1b1cfe236b4" />

**Distributions des profils clients :**

<img width="904" height="481" alt="image" src="https://github.com/user-attachments/assets/bad7aac2-f129-4aac-bb76-f69ea7723a2f" />

**Synthèse :**

<img width="912" height="186" alt="image" src="https://github.com/user-attachments/assets/ee34c908-a298-4ccc-904e-b7adddb59c13" />

**La section de synthèse met en avant des constats clés, notamment :**

- Les clients inactifs présentent un taux de churn significativement plus élevé

- Le churn augmente après 40 ans, surtout avec peu de produits

- Certains pays présentent un risque de churn plus élevé

- Les clients possédant un seul produit sont les plus fragiles

## Remarques

XGBoost s’exécute automatiquement sur CPU ou GPU selon l’environnement.

⚠️ L’optimisation des hyperparamètres peut prendre plusieurs minutes selon le modèle et le matériel utilisé (GPU recommandé).

## 👤 Auteur

Ce projet a été développé par **Djebril Laouedj**,
étudiant en dernière année en **Big Data & Intelligence Artificielle** à l'**ECE Paris**.
