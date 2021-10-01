from abc import ABC, abstractmethod
from coopstructs.geometry import Rectangle
from coopstructs.vectors import IVector, Vector2
from coopgraph.dataStructs import GridPoint
from typing import Dict, List, Tuple, Any, Set
import numpy as np
from coopgraph.gridSelectPolicies import IOnGridSelectPolicy, DoNothingPolicy
from coopgraph.gridState import GridState
from pprint import pformat


class AGrid(ABC):
    def __init__(self,
                 nRows: int,
                 nColumns: int,
                 values: np.array = None,
                 default_state: Dict=None,
                 grid_select_policies: List[IOnGridSelectPolicy] = None,
                 # neighbor_handler: Callable[[GridState, GridState], None] = None
                 ):
        self.nRows = nRows
        self.nColumns = nColumns
        self.grid = np.array([[GridState(self, row, column, values[row][column]) if values is not None else GridState(self, row, column, state=default_state)
                               for column in range(self.nColumns)] for row in range(self.nRows)])

        if grid_select_policies is None:
            grid_select_policies = [DoNothingPolicy("DN")]

        self._grid_select_policies = grid_select_policies
        self.initialize_policies()

    def __str__(self):
        ret = f"<{self.nRows} x {self.nColumns}> of type {type(self.grid[0][0])}"
        ret += f"\n{pformat(self.grid)}"

        return ret

    def __getitem__(self, item):
        return self.grid[item]

    def __iter__(self):
        for row in range(0, self.nRows):
            for col in range(0, self.nColumns):
                yield (Vector2(row, col), self.grid[row][col])

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

    def at(self, row, column):
        return self.grid[row][column]

    def act_on_grid(self, row: int, column: int, policies:List[IOnGridSelectPolicy]):
        for policy in policies:
            if policy not in self._grid_select_policies:
                continue
            else:
                policy.act_on_gridstate(self.grid[row][column])

        return self.grid[row][column]

    def initialize_policies(self):
        for grid in self.grid_enumerator:
            for policy in self._grid_select_policies:
                policy.initialize_state(grid)

    def state_value_as_array(self, key):
        return np.array([[self.grid[row][column].state[key]
                            for column in range(self.nColumns)]
                            for row in range(self.nRows)])

    def _eqls_or_in(self, val, collection):
        if type(collection) in [Set, List]:
            return val in collection
        else:
            return val == collection

    def coords_at_condition(self, rules: List[Tuple[Any, Any]]):
        passes = []
        for ii in self:
            for rule in rules:
                val = ii[1].state.get(rule[0], None)
                if val and self._eqls_or_in(val, rule[1]):
                    passes.append(ii[0])
        return passes