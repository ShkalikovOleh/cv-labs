import unittest
from typing import Iterable, List

from cv.csp import CSPBase, Node, ace, solve_with_propagation


class TestCSP(CSPBase[Node[int], int]):

    def __init__(self) -> None:
        super().__init__()

        nodes = [Node({0, 1}) for _ in range(3)]
        self.__nodes = nodes
        self.__neighbours = {nodes[0]: {nodes[1], nodes[2]},
                             nodes[1]: {nodes[0], nodes[2]},
                             nodes[2]: {nodes[0], nodes[1]}}

    @property
    def nodes(self) -> List[Node[int]]:
        return self.__nodes

    def get_neighbours(self, node: Node[int]) -> Iterable[Node[int]]:
        return self.__neighbours[node]

    def check_constraint(self, A: Node[int], B: Node[int], valA: int, valB: int):
        if A == self.__nodes[0] and valA == 1:
            return False
        if B == self.__nodes[0] and valB == 1:
            return False
        if A == self.__nodes[1] and B == self.__nodes[2] and valA == 1:
            return False
        if B == self.__nodes[1] and A == self.__nodes[2] and valB == 1:
            return False
        return True


class SolverTests(unittest.TestCase):

    def test_ace(self):
        csp = TestCSP()
        csp = ace(csp)

        self.assertEqual(csp.nodes[0].value, 0)
        self.assertEqual(csp.nodes[1].value, 0)
        self.assertFalse(csp.nodes[2].is_assigned())

    def test_solve_test_csp(self):
        csp = TestCSP()
        csp = solve_with_propagation(csp)

        self.assertTrue(csp.is_solved())
        self.assertEqual(csp.nodes[0].value, 0)
        self.assertEqual(csp.nodes[1].value, 0)
        self.assertIn(csp.nodes[2].value, {0, 1})
