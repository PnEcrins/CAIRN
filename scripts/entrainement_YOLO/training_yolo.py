# training_yolo26.py
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--data",      type=str, default="/home/weksteenl-ext/MyPython/datasets/yamls/tentes_tiled_01.yaml", help="Chemin vers le fichier YAML")
parser.add_argument("--name",      required=True)
parser.add_argument("--epochs",    type=int, default=3)
parser.add_argument("--freeze",    type=int, default=0)
parser.add_argument("--imgsz",     type=int, default=640)
parser.add_argument("--patience",  type=int, default=30)
parser.add_argument("--project", default="runs/detect")
parser.add_argument("--no-amp", action="store_false", dest="amp")
parser.add_argument("--batch", type=int, default= 16)
args = parser.parse_args()


model = YOLO("yolo26m.pt")

results = model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    freeze=args.freeze,
    patience=args.patience,
    project=args.project,
    name=args.name,
    amp=args.amp,
    batch=args.batch
)
