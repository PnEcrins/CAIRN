from marshmallow import Schema, ValidationError, fields
from marshmallow.fields import Field
import yaml


class PathField(Field):
    def _deserialize(self, value, attr, data, **kwargs):
        import os

        if not isinstance(value, str):
            raise ValidationError("Le chemin doit être une chaîne de caractères.")
        if not os.path.exists(value):
            raise ValidationError(f"Le chemin spécifié '{value}' n'existe pas.")
        return value

    def validate(self, value):
        import os

        if not isinstance(value, str):
            raise ValidationError("Le chemin doit être une chaîne de caractères.")
        if not os.path.exists(value):
            raise ValidationError(f"Le chemin spécifié '{value}' n'existe pas.")


class LogosSchema(Schema):
    top_left_1 = fields.Str(required=False, load_default=None)
    top_left_2 = fields.Str(required=False, load_default=None)
    top_right_1 = fields.Str(required=False, load_default=None)
    top_right_2 = fields.Str(required=False, load_default=None)


class UISchema(Schema):
    title = fields.Str(required=False, load_default="Détection et comptage automatique")
    theme_color = fields.Str(required=False, load_default="#981d97")
    page_threshold = fields.Int(required=False, load_default=100)
    show_visualization = fields.Bool(required=False, load_default=True)
    logos = fields.Nested(LogosSchema, required=False, load_default=dict)


class ModelsSchema(Schema):
    available = fields.List(fields.Str(), required=False, load_default=list)
    device = fields.Str(required=False, load_default="cpu")
    default_confidence = fields.Float(required=False, load_default=0.4)
    show_confidence_slider = fields.Bool(required=False, load_default=True)
    confidence_range = fields.List(fields.Float(), required=False, load_default=lambda: [0.1, 0.9])
    YOLO_MODEL_FILEPATH = PathField(required=False, load_default=None)
    SAM3_MODEL_FILEPATH = PathField(required=False, load_default=None)


class FeaturesSchema(Schema):
    allow_tiling = fields.Bool(required=False, load_default=True)
    classes = fields.List(fields.Str(), required=False, load_default=list)
    model_path = fields.Dict(
        keys=fields.Str(), values=fields.Str(), required=False, load_default=dict
    )


class OFBAttendanceSchema(Schema):
    ftp_server = fields.Str(required=False, load_default="")
    ftp_username = fields.Str(required=False, load_default="username")
    ftp_port = fields.Int(required=False, load_default=21)
    ftp_password = fields.Str(required=False, load_default="")
    ftp_directory = fields.Str(required=False, load_default="")
    local_folder = fields.Str(required=False, load_default="")
    output_folder = fields.Str(required=False, load_default="")
    model_name_pose = fields.Str(required=False, load_default="yolov8m-pose.pt")
    treshold_pose = fields.Float(required=False, load_default=0.3)
    model_name_google = fields.Str(required=False, load_default="yolov8m-oiv7.pt")
    treshold_google = fields.Float(required=False, load_default=0.1)
    image_or_time_csv = fields.Str(required=False, load_default="time")
    sequence_duration = fields.Int(required=False, load_default=10)
    time_step = fields.Str(required=False, load_default="h")
    output_format = fields.Str(required=False, load_default="csv")
    blur = fields.Bool(required=False, load_default=False)


class ConfigSchema(Schema):
    ui = fields.Nested(UISchema, required=False, load_default=dict)
    models = fields.Nested(ModelsSchema, required=False, load_default=dict)
    features = fields.Nested(FeaturesSchema, required=False, load_default=dict)
    ofb_attendance = fields.Nested(OFBAttendanceSchema, required=False, load_default=dict)


class ConfigNamespace(dict):
    """Wraps a dict recursively to allow attribute access at every level."""

    def __init__(self, data: dict):
        super().__init__(data)
        for key, value in data.items():
            setattr(self, key, ConfigNamespace(value) if isinstance(value, dict) else value)

    def __repr__(self):
        return f"{self.__class__.__name__}({dict.__repr__(self)})"


class Config(dict):

    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
                config_validated = ConfigSchema().load(config_data)
                self.update(config_validated)
                for key, value in config_validated.items():
                    setattr(
                        self, key, ConfigNamespace(value) if isinstance(value, dict) else value
                    )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Le fichier de configuration '{self.config_file}' est introuvable."
            )

        except ValidationError as e:
            raise ValueError(
                f"Erreur de validation dans le fichier de configuration '{self.config_file}': {e.messages}"
            )
