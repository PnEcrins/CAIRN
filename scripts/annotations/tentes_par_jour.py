import pandas as pd

# 1. Charger le fichier CSV propre
df = pd.read_csv('Annotations_Oceane_with_counts_clean.csv')

# 2. Grouper par Lac et Date pour trouver le maximum
# On inclut année, mois, jour pour être sûr de bien séparer chaque journée
df_daily = df.groupby(['lac', 'annee', 'mois', 'jour'])['Tente'].max().reset_index()

# 3. Créer une colonne Date lisible (JJ/MM/AAAA)
df_daily['Date'] = (
    df_daily['jour'].astype(int).astype(str) + '/' + 
    df_daily['mois'].astype(int).astype(str) + '/' + 
    df_daily['annee'].astype(int).astype(str)
)

# 4. Organiser et renommer les colonnes pour que ce soit propre
resultat = df_daily[['Date', 'lac', 'Tente']].rename(
    columns={'lac': 'Lac', 'Tente': 'Max_Tentes'}
)

# 5. Trier par date (du plus ancien au plus récent)
resultat = resultat.sort_values(['Date'])

# 6. Sauvegarder en CSV
resultat.to_csv('Liste_Max_Tentes_Par_Jour.csv', index=False)
