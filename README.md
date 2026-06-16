# CrowdShore 

**CrowdShore** est une application web (Gradio) de **détection et de
comptage automatique de la fréquentation** sur les sites naturels de
montagne — lacs d'altitude, zones de bivouac — développée dans le cadre des
projets **PLOUF** et **BiodivTourAlps** (Parc national des Écrins / LECA).

Elle permet d'analyser des séries d'images (issues de caméras
*timelapse*) pour détecter et compter
automatiquement les **baigneurs**, les **tentes** ou autre classes relatives à la fréquentation, à l'aide de modèles
de vision par ordinateur (YOLO, SAM3).

## 📸 Introduction

![Interface CrowdShore](docs/images/screenshot_app.png)

L'interface permet de :
- importer un lot d'images,
- choisir un modèle de détection (YOLO ou SAM3),
- choisir les classes à détecter(tente ou baigneur pour Yolo (entrainé dessus), n'importe quel prompt textuel pour SAM3)
- adapater si on le souhaite les paramètres de smodèles (seuil de confiance + tilling)
- lancer l'analyse et consulter un tableau de résultats par image,
- visualiser les détections (bounding boxes) sur chaque image,
- exporter l'ensemble des résultats au format CSV.

➡️ Documentation utilisateur complète : voir
[`docs/guide_utilisateur.md`](docs/guide_utilisateur.md).

## 🚀 Get Started

### Prérequis

- Python ≥ 3.10
- [Git LFS](https://git-lfs.com/) (images d'exemple et poids du modèle YOLO entraîné)
- GPU CUDA recommandé pour SAM3

### Installation rapide

```bash
git lfs install
git clone https://github.com/PnEcrins/timelapse-frequentation/App_web.git
cd timelapse-frequentation
git lfs pull

python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate

pip install -e .
```

### Lancer l'application

```bash
cd timelapse-frequentation/App_web
python app.py
```

### Accéder à l'interface

L'application est servie en local par défaut sur :

```
http://127.0.0.1:7860
```

Pour un accès réseau ou un partage temporaire, voir la section
*« Accéder à l'interface »* de [`docs/installation.md`](docs/installation.md).

➡️ Guide pas-à-pas de l'interface : [`docs/guide_utilisateur.md`](docs/guide_utilisateur.md).

## 🧠 Modèles disponibles

| Modèle | Description | Source / article | Téléchargement des poids |
|---|---|---|---|
| **YOLO26** (fine-tuné ) | Détecteur rapide et léger, spécialisé sur les classes `baigneur` / `tente`. Tiling optionnel via SAHI. | [Documentation Ultralytics](https://docs.ultralytics.com/models/yolo26#overview) | Poids fine-tunés versionnés via Git LFS (`timelapse-frequentation/App_web/models/weights/`) à partir d'un modèle pré-entraîné sur le dataset COCO |
| **SAM3** (Segment Anything Model 3) | Modèle de segmentation/détection par concept (texte), classes prédéfinies ou prompt libre. Ajout d'une fonctionnalité de tilling. | [SAM 3 — docs Ultralytics](https://docs.ultralytics.com/models/sam-3/) | Accès sur demande puis téléchargement manuel via la [page Hugging Face facebook/sam3](https://huggingface.co/facebook/sam3) (fichier `sam3.pt`) |

➡️ Détails (classes, mapping, tiling, licences) : [`docs/modeles.md`](docs/modeles.md).

## 🎨 Paramètres

L'apparence de l'application (logos, couleurs, thème, seuils par défaut)
est centralisée en tête de `timelapse-frequentation/App_web/app.py`.

| Élément | Valeur |
|---|---|
| Couleur baigneur | 🔴 `(239, 51, 64)` |
| Couleur tente | 🟢 `(0, 122, 94)` |
| Couleur prompt libre | 🟠 `(244, 162, 97)` |
| Couleur principale (thème) | `#981d97` (magenta PNE) |
| Police | Inter |
| Seuil de confiance par défaut | `0.4` |
| Seuil de pagination par jour | `100` images |


## 📊 Benchmark

Performances indicatives (validation interne, jeu de données aérien/drone
des lacs de montagne — Anterne, Lauvitel, Muzelle, Pormenaz, Lauzon,
Brevent, Cornu, Jovet) :

| Modèle | Classe | mAP50 | mAP50-95 | Seuil conseillé |
|---|---|---|---|---|
| YOLO |`baigneur` | 0.85 | 0.65 | 0.4 |
| YOLO | `tente` | 0.85 | 0.65 | 0.4 |
| SAM3 | `baigneur` | 0.80 | — | 0.4 |
| SAM3 | `tente` | 0.75 | — | 0.4 |

> Ces chiffres sont indicatifs et dépendent fortement du jeu de données
> d'évaluation, des conditions de prise de vue (altitude, luminosité,
> résolution) et de l'activation ou non du tiling.

## 🙏 Crédits

- **Parc national des Écrins** — pilotage du projet, terrain, charte
  graphique.
- **LECA** (Laboratoire d'Écologie Alpine) — appui scientifique (Vincent Miele).
- **Projets PLOUF & BiodivTourAlps** — financement et cadre des travaux de
  suivi de la fréquentation.

## 📄 Licence

Ce projet est distribué sous licence **MIT** — voir [`LICENSE`](LICENSE).

Les dépendances tierces (notamment Ultralytics, en AGPL-3.0) restent
soumises à leurs propres licences : voir la section *Licence* de
[`docs/modeles.md`](docs/modeles.md).

## 📚 Documentation complète

Toute la documentation utilisateur est disponible dans le dossier
[`docs/`](docs/index.md) :

- [Installation détaillée](docs/installation.md)
- [Guide utilisateur](docs/guide_utilisateur.md)
- [Modèles disponibles](docs/modeles.md)
- [Architecture du projet](docs/architecture.md)

Voir aussi le [`CHANGELOG.md`](CHANGELOG.md) pour l'historique des versions.
