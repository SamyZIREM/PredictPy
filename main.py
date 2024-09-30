import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split

#Chargement des données
df = pd.read_csv('loan_data.csv')

#Vérifier les données manquantes (aucune variable nulle)
données_manquantes = df.isnull().sum()
#print(données_manquantes) -> on n'a pas de valeurs manquantes 

#Affichage graphique pour voir la répartition entre ceux qui ont remboursé et ceux qui ne l'ont pas fait
#On conclut que le nbr de personnes ayant remboursé dépasse largement ceux qui n'ont pas remboursé

plt.figure(figsize=(8, 6))
sns.countplot(x='not.fully.paid', data=df, palette="Set2")
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
sns.boxplot(x='not.fully.paid', y='int.rate', data=loan_data_encoded, palette="Set2")
plt.title('Relation entre le taux dintérêt et le remboursement')
plt.xlabel('Not Fully Paid (1 = Non remboursé, 0 = Remboursé)')
plt.ylabel('Taux dintérêt')
#plt.show()


#Séparation des caractéristiques et la cible (remboursé ou non)
X = loan_data_encoded.drop('not.fully.paid', axis=1)  # Caractéristiques
y = loan_data_encoded['not.fully.paid']  # Cible

#Division des données en set train et set test (80% entraînement et 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
