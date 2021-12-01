from typing import Generic, Set, TypeVar


DT = TypeVar('DT')  # domain type


class Node(Generic[DT]):

    def __init__(self, domain: Set[DT]) -> None:
        assert len(domain) > 0

        self._domain = domain

    @property
    def domain(self) -> Set[DT]:
        return self._domain

    def is_assigned(self) -> bool:
        return len(self._domain) == 1

    def assign(self, value: DT) -> None:
        if len(self._domain) != 1 and value in self._domain:
            self._domain = {value}
        else:
            raise ValueError('Value out of range or has already assigned')

    def unassign(self, domain: Set[DT]) -> None:
        if len(self._domain) == 1:
            self._domain = domain
        else:
            raise ValueError('Value does not assinged')

    @property
    def value(self) -> DT:
        if len(self._domain) == 1:
            return next(iter(self._domain))
        else:
            raise ValueError('Value does not assinged')
