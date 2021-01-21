from abc import ABC, abstractmethod
from coopstructs.geometry import Rectangle
from coopgraph.graphs import Graph, Node
from coopstructs.vectors import IVector, Vector2
from coopgraph.dataStructs import GridPoint
from typing import Dict, List, Callable
import numpy as np
from coopgraph.gridSelectPolicies import IOnGridSelectPolicy, DoNothingPolicy
from coopgraph.gridState import GridState
from pprint import pformat

class AGrid(ABC, Graph):

    def __init__(self,
                 nRows: int,
                 nColumns: int,
                 graph_dict_provider: Callable[[Dict[IVector, Node]],
                                               Dict[Node, List[Node]]],
                 values: np.array = None,
                 default_state: Dict=None,
                 grid_select_policies: List[IOnGridSelectPolicy] = None,
                 # neighbor_handler: Callable[[GridState, GridState], None] = None
                 ):
        self.pos_node_map = self._build_position_node_map(ncols=nColumns, nrows=nRows)
        graph_dict = graph_dict_provider(self.pos_node_map)
        Graph.__init__(self, graph_dict=graph_dict)
        self.nRows = nRows
        self.nColumns = nColumns
        self.grid = np.array([[GridState(self, row, column, values[row][column]) if values else GridState(self, row, column, state=default_state)
                               for column in range(self.nColumns)] for row in range(self.nRows)])

        if grid_select_policies is None:
            grid_select_policies = [DoNothingPolicy("DN")]

        self._grid_select_policies = grid_select_policies
        self.initialize_policies()

        # self._neighbor_handler = neighbor_handler

    def __str__(self):
        ret = f"<{self.nRows} x {self.nColumns}> of type {type(self.grid[0][0])}"
        ret += f"\n{pformat(self.grid)}"

        # for row in range(0, self.nRows):
        #     ret = ret + f"\n{self.grid[row]}"


        return ret

    def __getitem__(self, item):
        return self.grid[item]


    @property
    def grid_enumerator(self):
        for row in range(0, self.nRows):
            for col in range(0, self.nColumns):
                yield self.grid[row][col]

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

    def path_between_points(self, point1: IVector, point2: IVector):

        if point1 not in self.pos_node_map or point2 not in self.pos_node_map:
            return None

        astar_ret = self.astar(self.pos_node_map[point1], self.pos_node_map[point2])
        if astar_ret.path is None:
            return None

        return [x.pos for x in astar_ret.path]

    def at(self, row, column):
        return self.grid[row][column]

    def _build_position_node_map(self, ncols: int, nrows: int):
        pos_node_map = {}
        for col in range(0, ncols):
            for row in range(0, nrows):
                pos = Vector2(row, col)
                pos_node_map[pos] = Node(str(pos), pos)

        return pos_node_map

    def act_on_grid(self, row: int, column: int, policies:List[IOnGridSelectPolicy]):
        # old_state = self.grid[row][column].copy()

        for policy in policies:
            if policy not in self._grid_select_policies:
                continue
            else:
                policy.act_on_gridstate(self.grid[row][column])
        # self.handle_others(old_state, self.grid[row][column])

    def initialize_policies(self):
        for grid in self.grid_enumerator:
            for policy in self._grid_select_policies:
                policy.initialize_state(grid)

    def state_value_as_array(self, key):
        return np.array([[self.grid[row][column].state[key]
                            for column in range(self.nColumns)]
                            for row in range(self.nRows)])


    #
    # def handle_others(self, old_state, new_state):
    #     if self._neighbor_handler is not None:
    #         self._neighbor_handler(old_state, new_state)

