import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import logging



#Chargement des données
df = pd.read_csv('loan_data.csv')
#print(df.head())

#Vérifier les données manquantes (aucune variable nulle)
données_manquantes = df.isnull().sum()
#print(données_manquantes) -> on n'a pas de valeurs manquantes 

#Affichage graphique pour voir la répartition entre ceux qui ont remboursé et ceux qui ne l'ont pas fait
#On conclut que le nbr de personnes ayant remboursé dépasse largement ceux qui n'ont pas remboursé

plt.figure(figsize=(8, 6))
sns.countplot(x='not.fully.paid', data=df, hue='not.fully.paid', legend=False, palette="Set2")
plt.title('Distribution des emprunteurs ayant remboursé ou non')
plt.xlabel('Not Fully Paid (1 = Non remboursé, 0 = Remboursé)')
plt.ylabel('Nombre d\'emprunteurs')
#plt.show()

#Conversion des variables catégorielles en numérique (dans notre cas purpose : représente l'objet du prêt)
encoder = OrdinalEncoder(cols=['purpose'])
loan_data_encoded = encoder.fit_transform(df)
#print(loan_data_encoded.head())

#Relation entre le taux d'interet et remboursement 
plt.figure(figsize=(8, 6))
sns.boxplot(x='not.fully.paid', y='int.rate', hue='not.fully.paid', data=loan_data_encoded, legend=False, palette="Set2")
plt.title('Relation entre le taux dintérêt et le remboursement')
plt.xlabel('Not Fully Paid (1 = Non remboursé, 0 = Remboursé)')
plt.ylabel('Taux dintérêt')
#plt.show()


#Séparation des caractéristiques et la cible (remboursé ou non)
X = loan_data_encoded.drop('not.fully.paid', axis=1)  # Caractéristiques
y = loan_data_encoded['not.fully.paid']  # Cible

#Division des données en set train et set test (80% entraînement et 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modèle de Régression Logistique
model = LogisticRegression(max_iter=5000, solver='liblinear')
model.fit(X_train_scaled, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the model: {accuracy * 100:.2f}%')



# Création de l'application Flask pour exposer le modèle en API
app = Flask(__name__)
# Configurez le logging
logging.basicConfig(level=logging.DEBUG)
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})
# Endpoint pour prédire si un emprunteur remboursera ou non son prêt
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.debug(f'Données reçues : {data}')
        
        # Créez un DataFrame à partir des données d'entrée
        df_new = pd.DataFrame(data, index=[0])
        logging.debug(f'DataFrame avant ajustement : {df_new}')

        # Remplir les colonnes manquantes avec des valeurs par défaut
        expected_columns = ['credit.policy', 'purpose', 'int.rate', 'installment', 'log.annual.inc', 
                            'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 
                            'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
        
        # Dictionnaire d'encodage pour la colonne 'purpose'
        purpose_mapping = {
            'credit_card': 1,
            'debt_consolidation': 2,
            'educational': 3,
            'major_purchase': 4,
            'small_business': 5,
            'all_other': 6
        }

        for col in expected_columns:
            if col not in df_new.columns:
                logging.debug(f'Colonne manquante : {col}')
                if col == 'installment':
                    try:
                        loan_amount = float(df_new['loan_amount'].values[0])
                        int_rate = float(df_new['int_rate'].values[0]) / 100
                        df_new[col] = (loan_amount * int_rate) / (1 - (1 + int_rate) ** -12)
                    except ValueError:
                        logging.error(f'Erreur de conversion pour loan_amount ou int_rate : {df_new["loan_amount"].values[0]}, {df_new["int_rate"].values[0]}')
                        return jsonify({'error': 'Invalid loan_amount or int_rate'}), 400

                elif col == 'log.annual.inc':
                    try:
                        loan_amount = float(df_new['loan_amount'].values[0])
                        if loan_amount > 0:
                            df_new[col] = np.log(loan_amount)
                        else:
                            df_new[col] = 0
                    except ValueError:
                        logging.error(f'Erreur de conversion pour loan_amount : {df_new["loan_amount"].values[0]}')
                        return jsonify({'error': 'Invalid loan_amount for log calculation'}), 400

                elif col == 'credit.policy':
                    df_new[col] = 1
                elif col == 'purpose':
                    # Encoder la valeur de 'purpose' après nettoyage
                    purpose_value = df_new['purpose'].values[0].strip()  # Supprimer les espaces
                    if purpose_value not in purpose_mapping:
                        logging.error(f'Valeur de purpose non reconnue : {purpose_value}')
                        return jsonify({'error': f'Unknown purpose: {purpose_value}'}), 400
                    df_new[col] = purpose_mapping[purpose_value]
                else:
                    df_new[col] = 0  # Valeur par défaut

        df_new = df_new[expected_columns]
        logging.debug(f'DataFrame après ajustement : {df_new}')

        # Effectuer la prédiction
        prediction = model.predict(df_new)
        logging.debug(f'Prédiction : {prediction}')

        # Convertir la prédiction en type standard Python
        prediction_result = int(prediction[0])  # Assurez-vous que prediction[0] est un type convertible
        return jsonify({'prediction': prediction_result})

    except Exception as e:
        logging.error(f'Erreur lors de la prédiction : {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
