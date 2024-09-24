import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # Import Matplotlib's pyplot

# Chargement des données
df = pd.read_csv('loan_data.csv')

# Vérifier les données manquantes (aucune variable nulle)
données_manquantes = df.isnull().sum()
#print(données_manquantes)

# Affichage graphique pour voir la répartition entre ceux qui ont remboursé et ceux qui ne l'ont pas fait
plt.figure(figsize=(8, 6))
sns.countplot(x='not.fully.paid', data=df, palette="Set2")
plt.title('Distribution des emprunteurs ayant remboursé ou non')
plt.xlabel('Not Fully Paid (1 = Non remboursé, 0 = Remboursé)')
plt.ylabel('Nombre d\'emprunteurs')
plt.show()

