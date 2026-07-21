Pipeline d'évaluation utilisée pour évaluer SAM3 sur cluster de calcul GPU (Bigfoot via GRICAD).

## Exemple d'utilisation sur un noeud de calcul

```bash
oarsub --name sam3_tiling -l /nodes=1/gpu=1,walltime=10:00:00 -p "gpumodel='A100'" --project pr-loupe --stdout "sam3_tiling_%jobid%.out" --stderr "sam3_tiling_%jobid%.err" "source /applis/environments/conda.sh && conda activate torch_2.11_cuda_12.2_with_ultralytics && cd ~/MyPython && for conf in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do conf_name=\$(echo \$conf | tr '.' '_'); python main.py --conf \$conf --output conf_tilling_baigneurs_prompt_swimmer\${conf_name}.csv --labels ./Annotations.json --images /bettik/PROJECTS/pr-loupe/weksteenl-ext --tiling --tile_ratio 0.3  --prompt \"swimmer\"; done"
```

Le fichier de sortie est un csv contenant les colonnes : image_name, image_id, lake_name, nb_preds, nb-gets, TP, FP, FN, conf, pred_boxes