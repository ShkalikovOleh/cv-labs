import unittest

from cv.csp import SudokuCSP


class SudokuCSPTests(unittest.TestCase):

    def test_n_should_be_greater_0(self):
        with self.assertRaises(AssertionError):
            SudokuCSP(0)

    def test_nodes(self):
        csp = SudokuCSP(2)
        nodes = iter(csp.nodes)
        expected_domain = {i for i in range(1, 5)}

        for i in range(1, 5):
            for j in range(1, 5):
                node = next(nodes)
                self.assertEqual(node.x, i)
                self.assertEqual(node.y, j)
                self.assertEqual(node.domain, expected_domain)

    def test_get_neighbours(self):
        csp = SudokuCSP(2)

        neighbours = csp.get_neighbours(csp.nodes[-1])

        n = 0
        expected_idxs = {(4, 3), (4, 2), (4, 1), (3, 3),
                         (3, 4), (2, 4), (1, 4)}
        for neigh in neighbours:
            n += 1
            idx = (neigh.x, neigh.y)
            self.assertTrue(idx in expected_idxs)

        self.assertEqual(n, 7)

    def test_check_constraint(self):
        csp = SudokuCSP(2)
        nodeA = csp.nodes[0]
        nodeB = csp.nodes[1]

        self.assertTrue(csp.check_constraint(nodeA, nodeA, 4, 3))
        self.assertTrue(csp.check_constraint(nodeA, nodeA, 3, 4))
        self.assertTrue(csp.check_constraint(nodeA, nodeB, 4, 3))
        self.assertTrue(csp.check_constraint(nodeB, nodeA, 3, 4))

        self.assertFalse(csp.check_constraint(nodeA, nodeA, 3, 3))
        self.assertFalse(csp.check_constraint(nodeA, nodeA, 4, 4))
        self.assertFalse(csp.check_constraint(nodeA, nodeB, 3, 3))
        self.assertFalse(csp.check_constraint(nodeB, nodeA, 4, 4))

    def test_is_solved(self):
        csp = SudokuCSP(1)
        self.assertTrue(csp.is_solved())

    def test_assign_point(self):
        csp = SudokuCSP(2)

        csp.assign_point(2, 2, 4)

        node = csp.nodes[5]
        self.assertTrue(node.is_assigned())
        self.assertEqual(node.value, 4)

    def test_assign_from_board(self):
        csp = SudokuCSP(2)
        board = [[1, 2, 3, 0],
                 [4, 3, 0, 0],
                 [0, 0, 0, 0],
                 [2, 1, 0, 4]]

        csp.assign_from_board(board)

        n = sum([node.is_assigned() for node in csp.nodes])
        self.assertEqual(n, 8)
        self.assertEqual(csp.nodes[-3].value, 1)
        self.assertFalse(csp.nodes[-2].is_assigned())

    def test_assign_from_board_should_raise_if_incorrect_board(self):
        csp = SudokuCSP(2)
        board1 = [[1, 2, 3, 0],
                  [4, 3, 0, 0],
                  [0, 0, 0, 0],
                  [2, 1, 0]]
        board2 = [[1, 2, 3, 0],
                  [4, 3, 0, 0],
                  [0, 0, 0, 0],
                  [2, 1, 0, -1]]
        board3 = [[1, 2, 3, 0],
                  [4, 3, 0, 0],
                  [0, 0, 0, 0]]

        with self.assertRaises(AssertionError):
            csp.assign_from_board(board1)
        with self.assertRaises(ValueError):
            csp.assign_from_board(board2)
        with self.assertRaises(AssertionError):
            csp.assign_from_board(board3)
