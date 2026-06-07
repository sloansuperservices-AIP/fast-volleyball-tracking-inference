from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
import numpy as np

@dataclass
class Detection:
    frame: int
    visibility: int
    x: float
    y: float
    radius: float = 20.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Frame": self.frame,
            "Visibility": self.visibility,
            "X": self.x,
            "Y": self.y,
            "Radius": self.radius
        }
