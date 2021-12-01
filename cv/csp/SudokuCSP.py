from typing import Iterable, List, Set

from .Node import Node
from .CSPBase import CSPBase


class SudokuNode(Node[int]):

    def __init__(self, x: int, y: int, domain: Set[int]) -> None:
        assert x > 0
        assert y > 0
        super().__init__(domain)

        self.__x = x
        self.__y = y

    @property
    def x(self) -> int:
        return self.__x

    @property
    def y(self) -> int:
        return self.__y


class SudokuCSP(CSPBase[SudokuNode, int]):

    def __init__(self, n: int) -> None:
        assert n > 0
        super().__init__()

        self.__n = n
        domain = {i for i in range(1, n**2 + 1)}
        self.__nodes = [SudokuNode(i, j, domain.copy())
                        for i in range(1, n**2 + 1)
                        for j in range(1, n**2 + 1)]

    def assign_from_board(self, board: List[List[int]]) -> None:
        n = self.__n
        assert len(board) == n**2
        assert all(len(row) == n**2 for row in board)

        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 0:
                    continue
                else:
                    self.__nodes[i * n**2 + j].assign(val)

    def assign_point(self, x: int, y: int, value: int) -> None:
        n = self.__n
        assert 0 < x <= n**2
        assert 0 < y <= n**2

        self.__nodes[(x - 1) * n**2 + y - 1].assign(value)

    @property
    def nodes(self) -> List[SudokuNode]:
        return self.__nodes

    def get_neighbours(self, node: SudokuNode) -> Iterable[SudokuNode]:
        n = self.__n
        for i in range(n**2):
            if i != node.x - 1:
                yield self.__nodes[i * n**2 + (node.y - 1)]
            if i != node.y - 1:
                yield self.__nodes[(node.x - 1) * n**2 + i]

        k = ((node.x - 1) // n) * n
        l = ((node.y - 1) // n) * n
        for i in range(k, k+n):
            for j in range(l, l+n):
                if i != node.x - 1 and j != node.y - 1:
                    yield self.__nodes[(i * n**2) + j]

    def check_constraint(self, A: SudokuNode, B: SudokuNode, valA: int, valB: int) -> bool:
        return valA != valB  # all different constraint

    def __repr__(self) -> str:
        n = self.__n
        result = f'Sudoku problem {n**2}x{n**2} \n'

        for i in range(n):
            result += ('|' + ('-' * (3*n**2 + n - 1)) + '|\n')
            for k in range(i * n, (i + 1) * n):
                for j in range(n ** 2):
                    if j % n == 0:
                        result += '|'
                    node = self.__nodes[k * n**2 + j]
                    result += f' {node.value if node.is_assigned() else " "} '
                result += '|\n'

        result += ('|' + '-' * (3*n**2 + n - 1) + '|\n')
        return result
