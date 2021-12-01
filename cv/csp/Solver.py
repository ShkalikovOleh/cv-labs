from typing import Callable
from copy import deepcopy

from .Node import Node
from .CSPBase import CSPBase


def ace(csp: CSPBase) -> CSPBase:
    '''
    Arc-Consistency Enforcing    
    '''

    csp = deepcopy(csp)
    queue = set(csp.nodes)

    while queue:
        node = queue.pop()
        val_for_remove = []

        for d in node.domain:
            neighbours = csp.get_neighbours(node)
            for neighbour in neighbours:
                # any(empty list) = False !!!
                if not any([csp.check_constraint(node, neighbour, d, nd) for nd in neighbour.domain]):
                    val_for_remove.append(d)
                    # because removing value from domain can break neighbours
                    for neigh in neighbours:
                        if len(neigh.domain) > 0:
                            queue.add(neigh)
                    break

        node.domain.difference_update(val_for_remove)

    return csp


def get_node_with_minimum_domain(csp: CSPBase) -> Node:
    result = None
    for node in csp.nodes[1:]:
        if not node.is_assigned():
            if result is None or len(result.domain) > len(node.domain):
                result = node

    if result is None:
        raise ValueError('CSP is inconsistent or solved')
    return result


def is_consistent(csp: CSPBase) -> bool:
    return all([len(node.domain) > 0 for node in csp.nodes])


def solve_with_propagation(csp: CSPBase, callback: Callable[[CSPBase], None] = None) -> CSPBase:
    result_csp = ace(csp)

    if csp.is_solved():
        return csp

    if not is_consistent(result_csp):
        raise ValueError('Constraints are inconsistent')

    while not result_csp.is_solved():
        curr_node = get_node_with_minimum_domain(result_csp)
        success = False
        domain = curr_node.domain.copy()
        for d in curr_node.domain:
            curr_node.assign(d)
            test_csp = ace(result_csp)

            if is_consistent(test_csp):
                success = True
                result_csp = test_csp
                if callback is not None:
                    callback(result_csp)
                break
            else:
                curr_node.unassign(domain)

        if not success:
            if callback is not None:
                callback(result_csp)
            raise ValueError('CSP does not have polymorphism')

    return result_csp
