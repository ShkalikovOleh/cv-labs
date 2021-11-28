from abc import ABC
from typing import Generic, Iterable, List, TypeVar

from .Node import DT, Node


NT = TypeVar('NT', bound=Node)  # node type


class CSPBase(Generic[NT, DT], ABC):

    @property
    def nodes(self) -> List[NT]:
        pass

    def get_neighbours(self, node: NT) -> Iterable[NT]:
        pass

    def check_constraint(self, A: NT, B: NT, valA: DT, valB: DT):
        pass

    def is_solved(self) -> bool:
        return all([node.is_assigned() for node in self.nodes])
