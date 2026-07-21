Ensemble des scripts utilisés pour fine-tuner un YOLO26 sur des images contenant des tentes et des baigneurs. Utilisation de la fonction train d'Ultralytics.

## Exemple d'utilisation sur un noeud de calcul

```bash

oarsub --name train        -l /nodes=1/gpu=1,walltime=10:00:00        -p "gpumodel='A100'"        --project pr-loupe        --stdout "train_%jobid%.out"        --stderr "train_%jobid%.err"        "source /applis/environments/conda.sh && conda activate torch_2.11_cuda_12.2_with_ultralytics && cd ~/MyPython && python training_yolo.py --epochs 300 --patience 50 --imgsz 640 --name tilling_allimage_tentes_baigneursfreeze --data /home/weksteenl-ext/MyPython/datasets/yamls/all_tentes_baigneurs.yaml --freeze 10"


```

La sortie est un dossier contenant les poids du modèle fine-tuner (le modèle "best" et le "last") et des fichiers .jpg contenant des visualisations des résultats et métriques.


Les données d'entraînemetn sont issues de la plateforme Labelstudio avec un export au format YOLO.