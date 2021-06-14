from typing import Dict, List
from coopstructs.vectors import Vector2
from coopstructs.geometry import Rectangle
from coopgraph.dataStructs import GridPoint
from coopgraph.AGrid import AGrid
from coopgraph.gridSelectPolicies import IOnGridSelectPolicy
import numpy as np

class RectGrid(AGrid):
    def __init__(self,
                 nRows: int,
                 nColumns: int,
                 grid_select_policies: List[IOnGridSelectPolicy] = None,
                 values: np.array = None,
                 default_state: Dict = None):
        AGrid.__init__(self,
                       nRows=nRows,
                       nColumns=nColumns,
                       grid_select_policies=grid_select_policies,
                       default_state=default_state,
                       values=values)


    def grid_unit_width(self, area_rect:Rectangle):
        return area_rect.width / self.nColumns

    def grid_unit_height(self, area_rect: Rectangle):
        return area_rect.height / self.nRows

    def coord_from_grid_pos(self, grid_pos: Vector2, area_rect: Rectangle, grid_point: GridPoint = GridPoint.CENTER) -> Vector2:
        offset = None
        if grid_point == GridPoint.CENTER:
            offset = 0.5
        elif grid_point == GridPoint.ORIGIN:
            offset = 0
        else:
            raise NotImplementedError(f"Unhandled grid point {grid_point}")
        grid_unit_width = self.grid_unit_width(area_rect)
        grid_unit_height = self.grid_unit_height(area_rect)

        ret = Vector2(grid_unit_width * (grid_pos.x + offset), grid_unit_height * (grid_pos.y + offset))
        return ret

    def grid_from_coord(self, coord: Vector2, area_rect: Rectangle) -> Vector2:
        if coord is None:
            return None

        # Change the x/y screen coordinates to grid coordinates
        column = int((coord.x) // (area_rect.width / self.nColumns))
        row = int((coord.y) // (area_rect.height / self.nRows))

        grid_coord = Vector2(column, row)
        return grid_coord

    def left_of(self, row: int, column: int, positions: int = 1):
        if column > positions:
            return self.grid[row][column - positions]
        else:
            return None

    def right_of(self, row: int, column: int, positions: int = 1):
        if column < self.nColumns - positions:
            return self.grid[row][column + positions]
        else:
            return None

    def up_of(self, row: int, column: int, positions: int = 1):
        if row > positions:
            return self.grid[row - positions][column]
        else:
            return None

    def down_of(self, row: int, column: int, positions: int = 1):
        if row < self.nRows - positions:
            return self.grid[row + positions][column]
        else:
            return None
