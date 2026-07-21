'''script permettant d'adapter le JSON  de sortie de l'outil Labelstudio pour qu'il soit comaptaible avec les chemins des images sur le cluster de calcul
'''


import json
import os


input_file = '/home/lweksteen/CrowdShore/Annotations/JSON/400.json' 
output_file = 'test_json.json'

with open(input_file, 'r') as f:
    data = json.load(f)

base_new_path = "/media/lweksteen/HDD 500 Go/Timelapses"

for img in data['images']:
    
    old_path = img['file_name']
    filename = os.path.basename(old_path)
    
    
    lac_name = filename.split('_')[0]
    
    
    new_path = os.path.join(base_new_path, lac_name, filename)
    
  
    img['file_name'] = new_path

with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Terminé ! Nouveau fichier créé : {output_file}")