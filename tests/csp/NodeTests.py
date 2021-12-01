import unittest

from cv.csp import Node
from cv.csp.SudokuCSP import SudokuNode


class NodeTests(unittest.TestCase):

    def test_domain_should_be_non_empty(self):
        with self.assertRaises(AssertionError):
            Node({})

    def test_assign_value(self):
        node = Node({1, 2, 3})
        node.assign(1)
        
        self.assertEqual(node.value, 1)
        self.assertEqual(node.domain, {1})
        self.assertTrue(node.is_assigned())
    
    def test_assign_value_via_domain(self):
        node = Node({1})

        self.assertEqual(node.value, 1)
        self.assertEqual(node.domain, {1})
        self.assertTrue(node.is_assigned())

    def test_sudoku_node_position_greater_than_0(self):
        domain = {0, 1}
        with self.assertRaises(AssertionError):
            node = SudokuNode(0, 1, domain)
        with self.assertRaises(AssertionError):
            node = SudokuNode(1, 0, domain)
        with self.assertRaises(AssertionError):
            node = SudokuNode(0, 0, domain)
        with self.assertRaises(AssertionError):
            node = SudokuNode(-1, 1, domain)
        with self.assertRaises(AssertionError):
            node = SudokuNode(1, -1, domain)

