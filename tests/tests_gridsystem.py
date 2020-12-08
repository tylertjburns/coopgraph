import unittest
from coopgraph.grids import GridSystem
from coopstructs.geometry import Rectangle
from coopstructs.vectors import Vector2
from coopgraph.dataStructs import GridPoint

class Tests_GridSystem(unittest.TestCase):

    def test_create_a_grid(self):
        grid = GridSystem(50, 100)

        assert grid.shape == (50, 100)

    def test_grid_size(self):
        grid = GridSystem(50, 100)
        rect = Rectangle(0, 0, 100, 300)

        grid_width = grid.grid_unit_width(rect)
        grid_height = grid.grid_unit_height(rect)

        assert grid_width == 3
        assert grid_height == 2

    def test_coord_from_grid_pos(self):
        grid = GridSystem(50, 100)
        rect = Rectangle(0, 0, 100, 300)

        x_pos = 50
        y_pos = 25
        grid_pos = Vector2(x_pos - 1, y_pos - 1)

        center_coord = grid.coord_from_grid_pos(grid_pos=grid_pos, area_rect=rect, grid_point=GridPoint.CENTER)
        origin_coord = grid.coord_from_grid_pos(grid_pos=grid_pos, area_rect=rect, grid_point=GridPoint.ORIGIN)

        grid_width = grid.grid_unit_width(rect)
        grid_height = grid.grid_unit_height(rect)

        assert center_coord == Vector2((x_pos - .5) * grid_width, (y_pos - .5) * grid_height)
        assert origin_coord == Vector2((x_pos - 1) * grid_width, (y_pos - 1) * grid_height)

    def test_grid_from_coord(self):
        grid = GridSystem(50, 100)
        rect = Rectangle(0, 0, 100, 300)

        x_coord = 148.5
        y_coord = 49
        coord_pos = Vector2(x_coord, y_coord)

        grid_pos = grid.grid_from_coord(coord=coord_pos, area_rect=rect)

        column = int((coord_pos.x) // (rect.width / grid.nColumns))
        row = int((coord_pos.y) // (rect.height / grid.nRows))

        assert grid_pos == Vector2(column, row)


    def test_astar(self):
        grid = GridSystem(10, 10)

        astar = grid.astar(grid.nodes_at_point(Vector2(0, 0))[0], grid.nodes_at_point(Vector2(5, 5))[0])


        assert astar.path == [grid.nodes_at_point(Vector2(0, 0))[0],
                              grid.nodes_at_point(Vector2(1, 1))[0],
                              grid.nodes_at_point(Vector2(2, 2))[0],
                              grid.nodes_at_point(Vector2(3, 3))[0],
                              grid.nodes_at_point(Vector2(4, 4))[0],
                              grid.nodes_at_point(Vector2(5, 5))[0]]