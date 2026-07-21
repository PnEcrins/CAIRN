import pandas as pd

df= pd.read_csv('Annotations_Oceane_with_counts_clean.csv')



Labels=['Tente','Baigneur']
#,'Bateau','Animal indeterminé','Randonneur','Autre animal','Paddle','Feu de camp','Véhicule motorisé']

for label in Labels:
    total = df[label].sum()
    print(f"Total {label}: {total}")


stats_lac = df.groupby('lac')[ 'Tente'].sum().reset_index()
print(stats_lac)

stats_lac = df.groupby('lac')[ 'Baigneur'].sum().reset_index()
print(stats_lac)


nb_img_tentes = (df['Tente'] > 0).sum()
nb_img_baigneurs = (df['Baigneur'] > 0).sum()

print(f"Nombre d'images avec au moins une tente : {nb_img_tentes}")
print(f"Nombre d'images avec au moins un baigneur : {nb_img_baigneurs}")


