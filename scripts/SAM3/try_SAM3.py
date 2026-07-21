from ultralytics.models.sam import SAM3SemanticPredictor

# Initialize predictor with configuration
overrides = dict(
    conf=0.4,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=False,  
    save=True,
)
predictor = SAM3SemanticPredictor(overrides=overrides)


predictor.set_image("/home/lweksteen/CrowdShore/Images test/Pormenaz_2024-08-13_14-05-01.jpg")

# Query with a single concept
results = predictor(text=["swimmer"])

# Query with a bouding box
#results = predictor(bboxes=[[227, 213, 239, 223]])