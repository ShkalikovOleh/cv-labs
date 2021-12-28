from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class GrammarBase(ABC):

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @abstractmethod
    def recognize_terminal(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_rules_for_dim(self, dim) -> np.ndarray:
        pass
