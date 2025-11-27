from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class InternalPoint:
    id: str               # uuid4 en string
    name: str             # "P1", "centro", etc.
    u: float              # coord. local-normalizada dentro del rect [0,1]
    v: float              # coord. local-normalizada dentro del rect [0,1]
    value: Optional[Tuple[float, ...]] = None  # valor de pixel (grayscale -> (v,), RGB -> (r,g,b))
    visible: bool = True
