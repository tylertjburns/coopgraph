from coopgraph.graphs import Node
from coopstructs.vectors import Vector2
import random as rnd
from typing import Callable, Tuple
import string
import math

def random_dict(index_provider: Callable[[], str], max_x: int, max_y: int):
    nodes = []
    for ii in range(0, 5):
        nodes.append(Node(name=f"{index_provider()}",
                          pos=(rnd.randint(0, max_x), rnd.randint(0, max_y))))

    graph_dict = {node: [] for node in nodes}
    for node in nodes:
        n_connections = rnd.randint(1, len(nodes)//2)
        samples = rnd.sample(nodes, n_connections)

        graph_dict[node] += [sample for sample in samples if sample not in graph_dict[node]]
        for sample in samples:
            graph_dict[sample].append(node)

    return graph_dict

def small_test():
    a = Node('a', (100, 100))
    b = Node('b', (200, 100))
    c = Node('c', (100, 200))
    d = Node('d', (200, 200))
    e = Node('e', (300, 300))
    graph_dict = {
        a: [b, c],
        b: [a, d, e],
        c: [a, d],
        d: [b, c, e],
        e: [b, d]
    }

    return graph_dict

def test_circuit():
    a = Node('a', (200, 200))
    b = Node('b', (250, 250))
    c = Node('c', (250, 300))
    d = Node('d', (300, 300))
    e = Node('e', (350, 150))
    f = Node('f', (300, 200))
    g = Node('g', (250, 100))
    h = Node('h', (200, 100))
    graph_dict = {
        a: [b, c],
        b: [c, d],
        c: [d],
        d: [e],
        e: [f, g],
        f: [h],
        h: [a],
        g: [a]
    }

    return graph_dict

def large_circuit(x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], spread: int = 100):
    alphabet=string.ascii_lowercase

    nodes = []
    rnd.seed(0)
    for ii in range(0, 26):
        nodes.append(
            Node(name=alphabet[ii],
                 pos=((x_bounds[1] - x_bounds[0]) / 2 + x_bounds[0] + (x_bounds[1] - x_bounds[0]) * math.sin(ii / 26 * 2 * math.pi) + rnd.randint(-spread, spread),
                     (y_bounds[1] - y_bounds[0]) / 2 + y_bounds[0] + (y_bounds[1] - y_bounds[0])  * math.cos(ii / 26 * 2 * math.pi) + rnd.randint(-spread, spread))))

    graph_dict = {}

    for ii, node in enumerate(nodes):
        n_connections = rnd.randint(1, 4)

        for n in range(1, n_connections + 1):
            if ii + n >= 26:
                index = ii + n - 26
            else:
                index = ii + n
            graph_dict.setdefault(node, []).append(nodes[index])
    return graph_dict


def basic_intersection():
    a = Node('a', (200, 200))
    b = Node('b', (200, 300))
    c = Node('c', (300, 300))
    d = Node('d', (300, 300))
    e = Node('e', (350, 150))
    f = Node('f', (300, 200))
    g = Node('g', (250, 100))
    h = Node('h', (200, 100))
    graph_dict = {
        a: [b, c],
        b: [c, d],
        c: [d],
        d: [e],
        e: [f, g],
        f: [h],
        h: [a],
        g: [a]
    }

    return graph_dict