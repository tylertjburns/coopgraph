import unittest
from cooptools.sectors.grids import RectGrid
from coopgraph.gridgraph import GridGraph
from coopstructs.vectors import Vector2

class Tests_GridGraph(unittest.TestCase):

    def test_astar(self):
        grid = RectGrid(10, 10)
        grid_graph = GridGraph(grid=grid)

        astar = grid_graph.astar_between_grid_pos(Vector2(0, 0), Vector2(5, 5))

        assert astar.path == [grid_graph.graph.nodes_at_point(Vector2(0, 0))[0],
                              grid_graph.graph.nodes_at_point(Vector2(1, 1))[0],
                              grid_graph.graph.nodes_at_point(Vector2(2, 2))[0],
                              grid_graph.graph.nodes_at_point(Vector2(3, 3))[0],
                              grid_graph.graph.nodes_at_point(Vector2(4, 4))[0],
                              grid_graph.graph.nodes_at_point(Vector2(5, 5))[0]]