
from marshmallow import Schema, ValidationError
from marshmallow.fields import Field
import tomllib

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

class ConfigSchema(Schema):
    YOLO_MODEL_FILEPATH = PathField(required=False,load_default=None)
    SAM3_MODEL_FILEPATH = PathField(required=False,load_default=None)

class Config(dict):

    def __init__(self, config_file="config.toml"):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, "rb") as f:
                config_data = tomllib.load(f)
                config_validated =ConfigSchema().load(config_data)
                self.update(config_validated)  # Validation
                for key, value in config_validated.items():
                    setattr(self, key, value)  # Set attributes for easy access
        except FileNotFoundError:
            raise FileNotFoundError(f"Le fichier de configuration '{self.config_file}' est introuvable.")
    
        except ValidationError as e:
            raise ValueError(f"Erreur de validation dans le fichier de configuration '{self.config_file}': {e.messages}")
        
