# Guide utilisateur — Détection et Comptage Automatique

Application web locale permettant d'analyser des images de timelapse ou de pièges photos afin de détecter et compter automatiquement des baigneurs et des tentes au bord des lacs de montagne.

---

## Sommaire

1. [Interface générale](#1-interface-générale)
2. [Étape 1 — Importer les images](#2-étape-1--importer-les-images)
3. [Étape 2 — Paramètres de l'IA](#3-étape-2--paramètres-de-lia)
4. [Lancer et annuler l'analyse](#4-lancer-et-annuler-lanalyse)
5. [Résultats et visualisation](#5-résultats-et-visualisation)
6. [Export CSV](#6-export-csv)
7. [Configuration avancée](#7-configuration-avancée)



## 1. Interface générale

![Aperçu de l'application](images/screenshot_app.png)

L'interface est divisée en deux colonnes :

| Zone | Contenu |
|---|---|
| **Colonne gauche** | Import des images, paramètres IA, boutons d'action |
| **Colonne droite** | Tableau des résultats et visualisation des détections |

---

## 2. Étape 1 — Importer les images

### Mode « Upload fichiers »

Cliquez sur la zone d'import pour sélectionner une ou plusieurs images depuis votre ordinateur (formats acceptés : `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.bmp`).  
Un compteur confirme le nombre d'images chargées.

### Mode « Dossier local »

Saisissez le chemin absolu d'un dossier contenant vos images (ex. `/home/user/mes_photos`).  
Les images sont lues directement depuis le disque, sans copie.

> **Basculer entre les modes** : utilisez le sélecteur **Mode d'import des images** en haut du panneau gauche.

---

## 3. Étape 2 — Paramètres de l'IA

### Type d'analyse

| Option | Description |
|---|---|
| **Analyse Timelapse** | Détection de tentes et baigneurs sur des photos de lac prises par timelapse. |
| **Analyse Piège Photos Randonneurs** | Comptage et classification de randonneurs sur des photos de pièges photographiques. Requiert des poids de modèle spécifiques. |

---

### Modèles disponibles (Timelapse uniquement)

#### YOLO *(recommandé)*

Modèle léger et rapide, entraîné sur 3 000 images de lacs de montagne.  
Optimisé pour la détection de tentes prises de loin.  
**Seuil conseillé : 0,4**

#### SAM3

Modèle polyvalent capable de détecter n'importe quelle catégorie via un **prompt libre**.  
⚠️ Très gourmand en ressources — un GPU est fortement recommandé (plusieurs dizaines de secondes par image en CPU).  
**Seuil conseillé : 0,4**

---

### Cibles à rechercher

Cochez les catégories à détecter :

- **Baigneur** — personnes se baignant
- **Tente** — tentes installées au bord du lac

Plusieurs cibles peuvent être sélectionnées simultanément.

### Prompt libre *(SAM3 uniquement)*

Saisissez librement la catégorie d'objet à rechercher (ex. `sac à dos`, `chien`).  
Les résultats apparaîtront dans une colonne `count_prompt` dans le CSV.

### Seuil de confiance

Curseur entre 0,1 et 0,9 contrôlant la sensibilité de la détection.  
Un seuil bas augmente le nombre de détections (au risque de faux positifs).  
Un seuil élevé n'affiche que les détections les plus certaines.

### Tiling (petits objets)

Option permettant de découper l'image en tuiles avant analyse, améliorant la détection des objets éloignés ou de petite taille.  
**Fortement recommandé** pour les vues de lac avec des sujets lointains.

---

## 4. Lancer et annuler l'analyse

| Bouton | Action |
|---|---|
| **Lancer l'analyse** | Démarre le traitement de toutes les images sélectionnées. |
| **Annuler** | Interrompt l'analyse en cours. |

La barre de statut sous les boutons indique la progression et le résumé final (nombre d'images traitées, nombre d'objets détectés).

---

## 5. Résultats et visualisation

Une fois l'analyse terminée, la colonne droite affiche le **tableau des résultats**.

### Filtrer par date

Si les images contiennent des métadonnées EXIF de date et que le volume dépasse le seuil configuré, un menu déroulant **Filtrer par date** permet d'afficher les résultats jour par jour.

### Visualiser une détection

Cliquez sur n'importe quelle ligne du tableau pour afficher :

- L'image correspondante avec les **boîtes de détection** dessinées (rouge = baigneur, vert = tente, orange = prompt libre).
- Un résumé textuel sous l'image (nom du fichier, comptes par catégorie).

---

## 6. Export CSV

Un fichier CSV est généré automatiquement à la fin de chaque analyse. Cliquez sur le lien **Enregistrer le rapport (CSV)** pour le télécharger.

### Structure du fichier (Timelapse)

| Colonne | Description |
|---|---|
| `image_name` | Nom du fichier image |
| `datetime` | Date et heure extraites des métadonnées EXIF |
| `year` / `month` / `day` / `hour` | Composantes temporelles |
| `count_baigneur` | Nombre de baigneurs détectés |
| `bbox_baigneur` | Coordonnées des boîtes englobantes (baigneurs) |
| `count_tente` | Nombre de tentes détectées |
| `bbox_tente` | Coordonnées des boîtes englobantes (tentes) |
| `count_prompt` | Nombre de détections via prompt libre |
| `bbox_prompt` | Coordonnées des boîtes englobantes (prompt) |

> Les colonnes absentes des cibles sélectionnées ne sont pas incluses dans le fichier exporté.

### Structure du fichier (Piège Photos)

Le CSV produit par l'analyse piège photos reprend le format natif du script `ofb_attendance` (classification par séquence et agrégation temporelle).


```
