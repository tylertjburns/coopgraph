import unittest
from coopgraph.grids import RectGrid
from coopstructs.geometry import Rectangle
from coopstructs.vectors import Vector2
from coopgraph.dataStructs import GridPoint
import time

class Tests_GridSystem(unittest.TestCase):

    def test_create_a_grid(self):
        grid = RectGrid(50, 100)

        assert grid.shape == (50, 100)

    def test_grid_size(self):
        grid = RectGrid(50, 100)
        rect = Rectangle(0, 0, 100, 300)

        grid_width = grid.grid_unit_width(rect)
        grid_height = grid.grid_unit_height(rect)

        assert grid_width == 3
        assert grid_height == 2


    def test_create_big_grid_100x100(self):
        tic = time.perf_counter()
        grid = RectGrid(100, 100)
        toc = time.perf_counter()

        self.assertLess(toc - tic, 1, msg=f"It took {toc-tic} seconds to create a grid of {grid.nRows}x{grid.nColumns}")

    def test_create_big_grid_1000x1000(self):
        tic = time.perf_counter()
        grid = RectGrid(1000, 1000)
        toc = time.perf_counter()

        self.assertLess(toc - tic, 10, msg=f"It took {toc-tic} seconds to create a grid of {grid.nRows}x{grid.nColumns}")

    def test_coord_from_grid_pos(self):
        grid = RectGrid(50, 100)
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
        grid = RectGrid(50, 100)
        rect = Rectangle(0, 0, 100, 300)

        x_coord = 148.5
        y_coord = 49
        coord_pos = Vector2(x_coord, y_coord)

        grid_pos = grid.grid_from_coord(coord=coord_pos, area_rect=rect)

        column = int((coord_pos.x) // (rect.width / grid.nColumns))
        row = int((coord_pos.y) // (rect.height / grid.nRows))

        assert grid_pos == Vector2(column, row)



