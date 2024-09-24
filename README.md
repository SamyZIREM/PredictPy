# Données sur les prêts
classer et prédire si l’emprunteur a remboursé ou non la totalité de son prêt.

# Étapes de réalisation :
1) Chragement des données
2) Vérification des valeurs manquantes
3) Analyse de données :
  graphe de stats : exemple : Affichage graphique pour voir la répartition entre ceux qui ont remboursé et ceux qui ne l'ont pas fait :

    ![image](https://github.com/user-attachments/assets/380dddc2-f5cb-40d5-97cd-bed550991979)

   La majorité des emprunteurs ont remboursé leur prêt en totalité (classe 0) : La barre représentant ceux qui ont remboursé leur prêt est beaucoup plus haute, indiquant un grand nombre d'emprunteurs dans cette 
   catégorie.
   Une petite proportion n'a pas remboursé en totalité (classe 1) : La barre pour ceux qui n'ont pas remboursé la totalité de leur prêt est nettement plus petite.

    Ci-dessous un boxplot pour la relation entre le taux d'intérêt et le statut de remboursement : 

    ![image](https://github.com/user-attachments/assets/e861df6f-b167-4eac-9973-ca16bd310216)

    Il semble que les taux d'intérêt soient en moyenne plus élevés pour les emprunteurs qui n'ont pas remboursé la totalité de leur prêt (classe 1) par rapport à ceux qui l'ont remboursé (classe 0).
    Cela indique une tendance où des taux d'intérêt plus élevés pourraient être associés à un risque accru de non-remboursement.

5) Conversion des variables catégorielles en numérique (dans notre cas purpose : représente l'objet du prêt) comme ceci vu que c'est une variable catégorielle nominale :
    "credit_card" → 1
    "debt_consolidation" → 2
    "educational" → 3
    "major_purchase" → 4
    "small_business" → 5
    "all_other" → 6
6) 
