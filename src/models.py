from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    cls_id: int
    radius: Optional[float] = None

@dataclass
class Point:
    x: float
    y: float
    frame: int
    radius: Optional[float] = None
