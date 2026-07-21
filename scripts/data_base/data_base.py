import pandas as pd
import os
import exifread
from PIL import Image

SOURCE = "/media/lweksteen/HDD 500 Go/Timelapses/"

colonnes = []

for dossier, _, fichiers in os.walk(SOURCE):
    for fichier in fichiers:
        if fichier.endswith(".jpg") or fichier.endswith(".JPG"):
            chemin_complet = os.path.join(dossier, fichier)
            chemin_relatif = os.path.relpath(chemin_complet, SOURCE)
            lac = chemin_relatif.split("/")[0]
            taille = os.path.getsize(chemin_complet) / (1024 * 1024)

            annee = mois = jour = heure = minute = seconde = "Unknown"
            date_str = None

            if lac == "Muzelle":
                img = Image.open(chemin_complet)
                exif = img.getexif()
                exif_ifd = exif.get_ifd(34665)
                date_str = exif_ifd.get(36867) or exif.get(36867) or exif.get(306)
                annee=date_str[0:4]
                mois=date_str[5:7]
                jour=date_str[8:10]
                heure=date_str[11:13]
                minute=date_str[14:16]
                seconde=date_str[17:19]   
            else:
                with open(chemin_complet, 'rb') as f:
                    tags = exifread.process_file(f, stop_tag='EXIF DateTimeOriginal')
                    date = tags.get('EXIF DateTimeOriginal')
                    if date is not None:
                        date_str = str(date)
                        annee=date_str[0:4]
                        mois=date_str[5:7]
                        jour=date_str[8:10]
                        heure=date_str[11:13]
                        minute=date_str[14:16]
                        seconde=date_str[17:19]                   

            colonnes.append({
                "lac":           lac,
                "chemin_relatif": chemin_relatif,
                "annee":         annee,
                "mois":          mois,
                "jour":          jour,
                "heure":         heure,
                "minute":        minute,
                "seconde":       seconde,
                "taille_mo":     round(taille, 2)
            })

df = pd.DataFrame(colonnes)
df.to_csv("metadata.csv", index=False)