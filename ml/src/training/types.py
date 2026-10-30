"""Tipos de arrays numéricos usados por entrenamiento e inferencia."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
