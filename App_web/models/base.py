"""
Classe de base abstraite pour tous les modèles de détection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Detection:
    """Un objet détecté dans une image."""
    image_name: str
    label: str          # ex: "baigneur", "tente"
    bbox: list[float]   # [x1, y1, x2, y2] en pixels
    confidence: float
    year: int | None
    month: int | None
    day: int | None
    hour: int | None


class BaseModel(ABC):

    @abstractmethod
    def detect(self, image_path: str, targets: list[str]) -> list[Detection]:
        """
        Détecte les objets dans une image.

        Args:
            image_path: chemin vers l'image
            targets: liste des classes à détecter, ex: ["baigneur", "tente"]

        Returns:
            liste de Detection (peut être vide)
        """
        ...