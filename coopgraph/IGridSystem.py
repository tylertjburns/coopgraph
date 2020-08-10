from abc import ABC, abstractmethod
from coopstructs.geometry import Rectangle
from coopgraph.graphs import Graph, Node
from coopstructs.vectors import IVector, Vector2
from coopgraph.dataStructs import GridPoint
from typing import Dict
import numpy as np

class IGridSystem(ABC, Graph):

    def __init__(self, nRows: int, nColumns: int, values: np.array = None, default_value=None, **kwargs):
        self.pos_node_map = self.build_position_node_map(ncols=nColumns, nrows=nRows)
        graph_dict = self.build_graph_dict(self.pos_node_map)
        Graph.__init__(self, graph_dict=graph_dict)
        self.nRows = nRows
        self.nColumns = nColumns
        self.grid = np.array([[values[x][y] if values else default_value for x in range(self.nColumns)] for y in range(self.nRows)])


    def __str__(self):
        ret = f"<{self.nRows} x {self.nColumns}> of type {type(self.grid[0][0])}"
        for x in range(0, self.nRows):
            ret = ret + f"\n{self.grid[x]}"

        return ret

    @property
    def shape(self):
        return self.grid.shape

    @abstractmethod
    def grid_unit_width(self, area_rect: Rectangle):
        pass

    @abstractmethod
    def grid_unit_height(self, area_rect: Rectangle):
        pass

    @abstractmethod
    def coord_from_grid_pos(self, grid_pos: Vector2, area_rect: Rectangle, grid_point: GridPoint = GridPoint.CENTER):
        pass

    @abstractmethod
    def grid_from_coord(self, coord: Vector2, area_rect: Rectangle):
        pass

    @abstractmethod
    def build_graph_dict(self, pos_node_map: Dict[IVector, Node], **kwargs):
        pass

    @abstractmethod
    def build_position_node_map(self, ncols: int, nrows: int):
        pass

    def path_between_points(self, point1: IVector, point2: IVector):

        if point1 not in self.pos_node_map or point2 not in self.pos_node_map:
            return None

        astar_ret = self.astar(self.pos_node_map[point1], self.pos_node_map[point2])
        if astar_ret is None:
            return None

        return [x.pos for x in astar_ret]

    def at(self, x, y):
        return self.grid[y][x]

    def build_position_node_map(self, ncols: int, nrows: int):
        pos_node_map = {}
        for x in range(0, ncols):
            for y in range(0, nrows):
                pos = Vector2(x, y)
                pos_node_map[pos] = Node(str(pos), pos)

        return pos_node_map