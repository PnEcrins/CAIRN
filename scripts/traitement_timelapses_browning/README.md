# Timelapse Process

Pipeline de traitement de vidéos timelapse (fichiers `.TLS`) : renommage, extraction d'images, et incrustation de l'horodatage dans les métadonnées EXIF via OCR. Script directement inspiré d'un code sous R: (https://github.com/tk-msu/TLS_file_extract)

Le script effectue 3 étapes :

1. **Renommage** des fichiers `.TLS` en `.avi`
2. **Extraction d'images** de chaque `.avi` avec `ffmpeg`
3. **OCR de l'horodatage** incrusté en bas de chaque image, puis écriture dans le champ EXIF `DateTimeOriginal` via `exiftool`

## Prérequis

### Outils système

- [ffmpeg](https://ffmpeg.org/) — extraction des images depuis les vidéos
- [exiftool](https://exiftool.org/) — écriture des métadonnées EXIF
- [tesseract-ocr](https://github.com/tesseract-ocr/tesseract) — moteur d'OCR

Sur Ubuntu/Debian :
```bash
sudo apt install ffmpeg exiftool tesseract-ocr
```

Sur macOS (Homebrew) :
```bash
brew install ffmpeg exiftool tesseract
```

Sur Windows, télécharger et installer chaque outil, puis vérifier qu'ils sont accessibles dans le PATH (ou adapter les chemins `FFMPEG_PATH` / `EXIFTOOL_PATH` dans le script).

### Dépendances Python

```bash
pip install pillow pytesseract
```

## Utilisation

```bash
python TLS_to_JPG.py /chemin/vers/le/dossier
```

Le dossier indiqué doit contenir les fichiers `.TLS` à traiter. Le script va :
- renommer chaque `.TLS` en `.avi` dans ce même dossier,
- créer un sous-dossier par vidéo (nommé comme la vidéo) contenant les images extraites (`images-0001.jpg`, `images-0002.jpg`, ...),
- pour chaque sous-dossier d'images, faire l'OCR de l'horodatage sur chaque image et l'inscrire dans les métadonnées EXIF.

## Gestion des échecs d'OCR : interpolation

L'OCR sur une bande d'incrustation vidéo n'est jamais fiable à 100 % (compression, flou de mouvement, luminosité variable d'une frame à l'autre). J'avais en effet un peu plus de 10% d'échec.
Plutôt que de laisser ces images sans date, le script exploite le fait que les images d'un même dossier sont extraites à intervalle régulier :

- Pour chaque image où l'OCR échoue, le script cherche les images valides les plus proches avant et après dans le même dossier.
- S'il en trouve une de chaque côté, il **interpole** linéairement l'horodatage entre les deux.
- Si l'échec est en début ou fin de séquence (pas d'image valide d'un des deux côtés), il **extrapole** à partir de l'intervalle moyen observé entre les dernières images valides connues.
- Ce n'est que si **aucune** image du dossier n'a pu être lue par l'OCR que le script abandonne : ces fichiers sont listés en console à la fin de l'exécution.

Ce mécanisme réduit considérablement le nombre d'images sans horodatage, sans avoir besoin d'un OCR parfait.

## Configuration à ajuster

Toutes les valeurs suivantes sont en haut du fichier `timelapse_process.py` :

| Paramètre | Rôle | Valeur par défaut |
|---|---|---|
| `FFMPEG_PATH` | Chemin vers l'exécutable ffmpeg | `"ffmpeg"` |
| `EXIFTOOL_PATH` | Chemin vers l'exécutable exiftool | `"exiftool"` |
| `FRAMERATE` | Nombre d'images extraites par seconde de vidéo | `5` |
| `CROP_BOX` | Zone de recadrage de la bande d'horodatage `(left, top, right, bottom)` | `(0, 1000, 1920, 1080)` |

**`FRAMERATE`** : à adapter selon le nombre d'images réellement contenues dans vos vidéos `.TLS` (ouvrez un fichier renommé en `.avi` dans un lecteur vidéo pour vérifier sa durée et son nombre d'images).

**`CROP_BOX`** : dépend du modèle de caméra et de la position de la bande d'incrustation (date/heure) sur l'image. La valeur par défaut correspond à une image 1920×1080 avec la bande dans les 80 derniers pixels en bas. À ajuster si vos images ont une résolution différente ou si la bande est positionnée ailleurs — utilisez `debug_ocr()` (voir plus bas) pour vérifier visuellement le recadrage.

## Débogage de l'OCR

Pour inspecter le résultat de l'OCR sur une image précise (par exemple une qui a échoué) :

```python
from timelapse_process import debug_ocr
from pathlib import Path

debug_ocr(Path("/chemin/vers/image.jpg"))
```

Cette fonction :
- sauvegarde la zone recadrée dans `test_strip.png` (dans le dossier courant) pour vérification visuelle du `CROP_BOX`,
- affiche le texte brut détecté par l'OCR,
- affiche le texte nettoyé et les correspondances trouvées par l'expression régulière.

Si `test_strip.png` ne montre pas correctement la bande d'horodatage, ajustez `CROP_BOX`. Si le crop est correct mais que le texte OCR est illisible sur certaines images, c'est normal ponctuellement (compression vidéo, flou) — c'est justement ce que l'interpolation compense.

## Notes

- Les fichiers pour lesquels ni l'OCR ni l'interpolation n'ont fonctionné (aucune image valide dans tout le dossier) sont listés en fin d'exécution dans la console.
- Le format d'horodatage attendu dans les images est `MM/JJ/AAAA HH:MM AM/PM` (avec ou sans espace avant AM/PM, avec ou sans espace entre la date et l'heure). Si votre caméra utilise un autre format (24h, JJ/MM/AAAA, etc.), il faudra adapter l'expression régulière `TIMESTAMP_REGEX` et la fonction `extract_timestamp()`.