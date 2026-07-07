# Guide Administrateur

Ce guide décrit la configuration de l'application via le fichier `config.yaml`.

---

## Structure du fichier `config.yaml`

Le fichier `config.yaml` est organisé en quatre sections principales : `ui`, `models`, `features` et `ofb_attendance`.

Un exemple de fichier de configuration est disponible dans `config.yaml.sample`.

---

## Section `ui` — Interface utilisateur

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `title` | chaîne | `"Détection et comptage automatique"` | Titre affiché dans l'interface. |
| `theme_color` | chaîne | `"#981d97"` | Couleur principale de l'interface (format hexadécimal). |
| `page_threshold` | entier | `100` | Nombre d'images par page dans la galerie. |
| `show_visualization` | booléen | `true` | Affiche ou masque le panneau de droite (tableaux et images). |
| `logos.top_left_1` | chaîne | _(aucun)_ | Chemin vers le logo en haut à gauche (position 1). |
| `logos.top_left_2` | chaîne | _(aucun)_ | Chemin vers le logo en haut à gauche (position 2). |
| `logos.top_right_1` | chaîne | _(aucun)_ | Chemin vers le logo en haut à droite (position 1). |
| `logos.top_right_2` | chaîne | _(aucun)_ | Chemin vers le logo en haut à droite (position 2). |

**Exemple :**
```yaml
ui:
  title: "Détection et comptage automatique"
  theme_color: "#981d97"
  page_threshold: 100
  show_visualization: true
  logos:
    top_left_1: "logo_ecrins.png"
    top_left_2: "logo_lacs.png"
    top_right_1: "logo_leca.png"
    top_right_2: "BiodivTourAlps_logo_def.png"
```

---

## Section `models` — Modèles de détection

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `available` | liste de chaînes | `[]` | Liste des modèles disponibles dans l'interface (ex. `YOLO`, `SAM3`). |
| `device` | chaîne | `"cpu"` | Dispositif d'inférence : `"cpu"` ou `"cuda"` (carte graphique). |
| `default_confidence` | flottant | `0.4` | Seuil de confiance appliqué par défaut. |
| `show_confidence_slider` | booléen | `true` | Affiche ou masque le curseur de confiance dans l'interface. Si `false`, la valeur par défaut est utilisée. |
| `confidence_range` | liste de 2 flottants | `[0.1, 0.9]` | Plage min/max du curseur de confiance. |
| `YOLO_MODEL_FILEPATH` | chemin | _(aucun)_ | Chemin vers le fichier de poids du modèle YOLO (doit exister sur le disque). |
| `SAM3_MODEL_FILEPATH` | chemin | _(aucun)_ | Chemin vers le fichier de poids du modèle SAM3 (doit exister sur le disque). |

> **Note :** Les champs `YOLO_MODEL_FILEPATH` et `SAM3_MODEL_FILEPATH` sont validés à la lecture de la configuration : le chemin spécifié doit exister sur le système de fichiers.

**Exemple :**
```yaml
models:
  available:
    - YOLO
    - SAM3
  device: "cpu"
  default_confidence: 0.4
  show_confidence_slider: true
  confidence_range: [0.1, 0.9]
```

---

## Section `features` — Fonctionnalités

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `allow_tiling` | booléen | `true` | Active ou masque entièrement l'option de tiling dans l'interface. |
| `classes` | liste de chaînes | `[]` | Classes d'objets détectables (ex. `tente`, `baigneur`). |
| `model_path` | dictionnaire | `{}` | Association entre le nom d'un modèle et le chemin de ses poids. |

**Exemple :**
```yaml
features:
  allow_tiling: true
  classes:
    - tente
    - baigneur
  model_path:
    YOLO: "model_weights/YOLO26_300epochs_tentes+baigneurs_10freeze.pt"
    SAM3: "model_weights/sam3.pt"
```

---

## Section `ofb_attendance` — Module de fréquentation OFB

Cette section configure le module de comptage automatique de fréquentation via timelapse (développé en partenariat avec l'OFB).

| Paramètre | Type | Défaut | Description |
|---|---|---|---|
| `ftp_server` | chaîne | `""` | Adresse du serveur FTP source des images. |
| `ftp_username` | chaîne | `"username"` | Nom d'utilisateur FTP. |
| `ftp_port` | entier | `21` | Port du serveur FTP. |
| `ftp_password` | chaîne | `""` | Mot de passe FTP. |
| `ftp_directory` | chaîne | `""` | Répertoire distant sur le serveur FTP contenant les images. |
| `local_folder` | chaîne | `""` | Dossier local où les images sont téléchargées. |
| `output_folder` | chaîne | `""` | Dossier local où les résultats sont écrits. |
| `model_name_pose` | chaîne | `"yolov8m-pose.pt"` | Nom du modèle de détection de pose utilisé. |
| `treshold_pose` | flottant | `0.3` | Seuil de confiance pour le modèle de pose. |
| `model_name_google` | chaîne | `"yolov8m-oiv7.pt"` | Nom du modèle de détection générique (Open Images). |
| `treshold_google` | flottant | `0.1` | Seuil de confiance pour le modèle générique. |
| `image_or_time_csv` | chaîne | `"time"` | Mode d'export CSV : `"time"` (par pas de temps) ou `"image"` (par image). |
| `sequence_duration` | entier | `10` | Durée d'une séquence en minutes. |
| `time_step` | chaîne | `"h"` | Pas de temps pour l'agrégation des résultats (`"h"` = heure, `"d"` = jour, etc.). |
| `output_format` | chaîne | `"csv"` | Format de sortie des résultats (ex. `"csv"`). |
| `blur` | booléen | `false` | Active le floutage des visages et des plaques dans les images de sortie. |

**Exemple :**
```yaml
ofb_attendance:
  ftp_server: "ftp.example.com"
  ftp_username: "monuser"
  ftp_port: 21
  ftp_password: "monmotdepasse"
  ftp_directory: "/images_timelapse"
  local_folder: "/data/images"
  output_folder: "/data/output"
  model_name_pose: "yolov8m-pose.pt"
  treshold_pose: 0.3
  model_name_google: "yolov8m-oiv7.pt"
  treshold_google: 0.1
  image_or_time_csv: "time"
  sequence_duration: 10
  time_step: "h"
  output_format: "csv"
  blur: false
```

