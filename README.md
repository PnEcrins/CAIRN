# CAIRN

**CAIRN** est une application web (Gradio) de **détection et de
comptage automatique de la fréquentation** sur les sites naturels de
montagne — lacs d'altitude, zones de bivouac — développée dans le cadre des
projets **PLOUF** et **BiodivTourAlps** (Parc national des Écrins / LECA).

Elle permet d'analyser des séries d'images (issues de caméras
*timelapse*) pour détecter et compter
automatiquement les **baigneurs**, les **tentes** ou autre classes relatives à la fréquentation, à l'aide de modèles
de vision par ordinateur (YOLO, SAM3).


![Interface CAIRN](docs/images/screenshot_app.png)

L'interface permet de :
- **Importer** un lot d'images à analyser
- **Choisir** d'analyser des images timelapses ou des détections automatiques (typiquement faite par des pièges photos). Cette dernière catégorie profite de l'ajout d'un projet antérieur : [OFB-Attendance](https://github.com/Attendance-PNE-OFB/yolov8-attendance)

Pour analyser des timelapses (cœur de l'application):

- **Choisir** un modèle de détection (YOLO ou SAM3)
- **Sélectionner** les classes à détecter (tente ou baigneur pour YOLO, prompt libre pour SAM3)
- **Ajuster** les paramètres du modèle (seuil de confiance et tiling optionnel)
- **Lancer** l'analyse et consulter les résultats dans un tableau détaillé par image
- **Visualiser** les détections (bounding boxes) directement sur les images
- **Exporter** l'ensemble des résultats au format CSV


Des images illustrant les capacités de détection du modèle sont proposées ici [`docs/images/exemples`](docs/images/exemples).

➡️ Documentation utilisateur complète : voir
[`docs/guide_utilisateur.md`](docs/guide_utilisateur.md).

## 🚀 Get Started

### Prérequis

- Python ≥ 3.10
- [Git LFS](https://git-lfs.com/) (images d'exemple et poids du modèle YOLO entraîné)
- GPU CUDA recommandé pour SAM3


### Cloner le dépôt et récupérer les poids du modèle YOLO

```bash
git lfs install
git clone https://github.com/PnEcrins/CAIRN.git
cd CAIRN
git lfs pull
```

### Créer un fichier de configuration `config.yaml` (exemple minimal) :

```bash
cp config.yaml.sample config.yaml
```

Plus d'informations sur la configuration : [`docs/guide_admin.md`](docs/guide_admin.md).

### Lancer l'application en local (hors Docker)

#### Créer un environnement virtuel et installer les dépendances
```bash
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate

pip install .
# ou uv sync
```

#### Lancer l'application

```bash
python app.py
```
L'application est servie en local par défaut sur http://127.0.0.1:7860


### Installation via Docker

```bash
docker-compose up -d
```

ou via l'image Docker officielle (GitHub Container Registry) :

```bash
docker run -d -p 80:7860 -v ./config.yaml:/app/config.yaml --name cairn ghcr.io/pnecrins/cairn:latest
```

➡️ Guide pas-à-pas de l'interface : [`docs/guide_utilisateur.md`](docs/guide_utilisateur.md).



##  Modèles disponibles

| Modèle | Description | Source / article | Téléchargement des poids |
|---|---|---|---|
| **YOLO26** (fine-tuné ) | Détecteur rapide et léger, spécialisé sur les classes `baigneur` / `tente`. Tiling optionnel via SAHI. | [Documentation Ultralytics](https://docs.ultralytics.com/models/yolo26#overview) | Poids fine-tunés versionnés via Git LFS (`CAIRN/models/weights/`) à partir d'un modèle pré-entraîné sur le dataset COCO |
| **SAM3** (Segment Anything Model 3) | Modèle de segmentation/détection par concept (texte), classes prédéfinies ou prompt libre. Ajout d'une fonctionnalité de tilling. | [SAM 3 — docs Ultralytics](https://docs.ultralytics.com/models/sam-3/) | Accès sur demande puis téléchargement manuel via la [page Hugging Face facebook/sam3](https://huggingface.co/facebook/sam3) (fichier `sam3.pt`) |

➡️ Détails (classes, mapping, tiling, licences) : [`docs/modeles.md`](docs/modeles.md).


### Installation du modèle SAM3

SAM3 est un modèle propriétaire (licence non commerciale) et n'est pas inclus dans le dépôt. Pour l'utiliser, il faut :
1. Créer un compte sur [Hugging Face](https://huggingface.co/).
2. Accepter les conditions d'utilisation du modèle SAM3 (licence non commerciale).
3. Télécharger le fichier `sam3.pt` depuis la page [facebook/sam3](https://huggingface.co/facebook/sam3).
4. Placer le fichier `sam3.pt` dans le dossier `CAIRN/model_weights/`.


> [!NOTE]
> Pour les utilisateurs de Docker, il est nécessaire de monter le fichier `sam3.pt` dans le conteneur via un volume Docker (voir exemple dans `docker-compose.yml`) ou d'exécuter la commande docker suivante avec le volume approprié :

```bash
docker run -d -p 80:7860 -v ./config.yaml:/app/config.yaml -v ./model_weights/sam3.pt:/app/model_weights/sam3.pt --name cairn ghcr.io/pnecrins/cairn:latest
``` 

## Configuration

L'application se configure via le fichier de configuration `config.yaml.sample`. Les paramètres sont répartis en plusieurs sections :

### Interface utilisateur (`ui`)
Règle l'apparence et l'affichage de l'application.

| Paramètre | Type | Valeur par défaut / Exemple | Description |
| :--- | :--- | :--- | :--- |
| `title` | Chaîne | `"CAIRN"` | Titre principal de l'application |
| `subtitle` | Chaîne | `"CAractérisation par..."` | Sous-titre explicatif |
| `theme_color` | Code hex | `"#981d97"` | Couleur principale du thème graphique |
| `page_threshold` | Entier | `100` | Seuil d'éléments affichés par page |
| `show_visualization` | Booleen | `true` | Affiche (`true`) ou masque (`false`) le panneau latéral droit (résultats et visuels) |
| `logos` | Dictionnaire | *(chemins d'images)* | Empacements des logos (`top_left_1`, `top_left_2`, `top_right_1`, `top_right_2`) |

### Modèles d'IA (`models`)
Gère les modèles disponibles et le matériel d'exécution.

| Paramètre | Type | Valeur par défaut / Exemple | Description |
| :--- | :--- | :--- | :--- |
| `available` | Liste | `[YOLO, SAM3]` | Liste des modèles sélectionnables dans l'application |
| `device` | Chaîne | `"cpu"` | Matériel d'exécution (`"cpu"` pour le processeur ou `"cuda"` pour le GPU) |
| `default_confidence` | Dictionnaire | `YOLO: 0.3`, `SAM3: 0.6` | Seuil de confiance par défaut pour chaque modèle |
| `show_confidence_slider` | Booleen | `true` | Affiche ou masque le curseur de réglage du seuil de confiance dans l'interface |
| `confidence_range` | Liste `[min, max]` | `[0.1, 0.9]` | Bornes minimale et maximale pour le curseur de confiance |

### Fonctionnalités & Poids (`features`)
Définit les options de traitement et la localisation des fichiers de poids.

| Paramètre | Type | Valeur par défaut / Exemple | Description |
| :--- | :--- | :--- | :--- |
| `allow_tiling` | Booleen | `true` | Autorise ou masque l'option de *tiling* (découpage d'image) |
| `classes` | Liste | `[tente, baigneur]` | Classes d'objets à détecter |
| `model_path` | Dictionnaire | *(chemins .pt)* | Chemin vers les fichiers de poids local pour chaque modèle (`YOLO`, `SAM3`) |

### Fréquentation OFB (`ofb_attendance`)
Paramètres liés à un autre projet : voir [le dépôt Git](https://github.com/Attendance-PNE-OFB/yolov8-attendance/blob/main/README-FR.md)


##  Benchmark

Performances indicatives (validation interne, jeu de données aérien/drone
des lacs de montagne — Anterne, Lauvitel, Muzelle, Pormenaz, Lauzon,
Brevent, Cornu, Jovet) :

| Modèle | Mode | mAP50 | Seuil conseillé |
|---|---|---|---|---|
| YOLO |Sans tiling | 0.5 |  0.3 |
| YOLO | Avec tiling | 0.85 | 0.3 |
| SAM3 | Sans tiling | 0.4 |  0.6 |
| SAM3 | Avec tiling| 0.5 |  0.5 |

> Ces chiffres sont indicatifs et dépendent fortement du jeu de données
> d'évaluation, des conditions de prise de vue (altitude, luminosité,
> résolution) et de l'activation ou non du tiling.

Au global, il est recommandé d'utiliser le modèle YOLO26 fine-tuné, beaucoup plus rapide en inférence que SAM3, mais aussi plus précis. 
On obtient de très bons résultats, notamment sur la détection de la classe "tente", beaucoup moins sur la classe "baigneur", faute de suffisamment de données d'entraînement. 

##  Crédits

- **Parc national des Écrins** — pilotage du projet, terrain, charte
  graphique.
- **LECA** (Laboratoire d'Écologie Alpine) — appui scientifique (Vincent Miele).
- **Projets PLOUF & BiodivTourAlps** — financement et cadre des travaux de
  suivi de la fréquentation.

##  Licence

Ce projet est distribué sous licence **AGPL-3.0** — voir [`LICENSE`](LICENSE).

##  Documentation complète

Toute la documentation utilisateur est disponible dans le dossier
[`docs/`](docs/) :


- [Guide utilisateur](docs/guide_utilisateur.md)
- [Guide administrateur](docs/guide_admin.md)

