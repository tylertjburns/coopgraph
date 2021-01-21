from coopgraph.graphs import Node
from typing import Dict, List, Callable
from coopstructs.vectors import Vector2, IVector
from coopstructs.geometry import Rectangle
from coopgraph.dataStructs import GridPoint, UnitPathDefinition
from coopgraph.AGrid import AGrid
from coopgraph.gridSelectPolicies import IOnGridSelectPolicy, TogglePolicy
from coopgraph.gridState import GridState
from coopgraph.toggles import BooleanToggleable

class GridSystem(AGrid):
    def __init__(self,
                 nRows: int,
                 nColumns: int,
                 connect_adjacents: bool=True,
                 connect_diagonals: bool=True,
                 grid_select_policies: List[IOnGridSelectPolicy] = None,
                 default_state: Dict = None):
        self._diagonal_connections = {}
        self._connect_adjacents = connect_adjacents
        self._connect_diagonals = connect_diagonals
        AGrid.__init__(self,
                       nRows=nRows,
                       nColumns=nColumns,
                       graph_dict_provider=self.graph_dict_provider,
                       grid_select_policies=grid_select_policies,
                       default_state=default_state)


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

    def graph_dict_provider(self, pos_node_map: Dict[IVector, Node]) -> Dict[Node, List[Node]]:
        return self.build_graph_dict(pos_node_map, connect_adjacents=self._connect_adjacents, connect_diagonals=self._connect_diagonals)

    def build_graph_dict(self,
                         pos_node_map: Dict[IVector, Node],
                         connect_adjacents: bool=True,
                         connect_diagonals: bool = True)-> Dict[Node, List[Node]]:
        graph_dict = {}

        for pos in pos_node_map.keys():
            graph_dict[pos_node_map[pos]] = []
            adjacents = [
                Vector2(pos.x - 1, pos.y) if pos_node_map.get(Vector2(pos.x - 1, pos.y), None) else None,  # left
                Vector2(pos.x + 1, pos.y) if pos_node_map.get(Vector2(pos.x + 1, pos.y), None) else None,  # right
                Vector2(pos.x, pos.y - 1) if pos_node_map.get(Vector2(pos.x, pos.y - 1), None) else None,  # up
                Vector2(pos.x, pos.y + 1) if pos_node_map.get(Vector2(pos.x, pos.y + 1), None) else None,  # down
            ]

            diagonals = [
                Vector2(pos.x - 1, pos.y - 1) if pos_node_map.get(Vector2(pos.x - 1, pos.y - 1), None) else None,
                # UpLeft
                Vector2(pos.x + 1, pos.y - 1) if pos_node_map.get(Vector2(pos.x + 1, pos.y - 1), None) else None,
                # UpRight
                Vector2(pos.x - 1, pos.y + 1) if pos_node_map.get(Vector2(pos.x - 1, pos.y + 1), None) else None,
                # DownLeft
                Vector2(pos.x + 1, pos.y + 1) if pos_node_map.get(Vector2(pos.x + 1, pos.y + 1), None) else None
                # DownRight
            ]

            connections = []

            # add adjacents to connections list
            if connect_adjacents:
                connections += adjacents

            # add diagonals to connections list
            if connect_diagonals:
                connections += diagonals

            # add diagonal connections to a saved dict for quick id later
            for connection in diagonals:
                if connection:
                    self._diagonal_connections.setdefault(pos, []).append(connection)

            # add connections to the graph_dict for each node
            for connection_pos in connections:
                try:
                    if connection_pos:
                        graph_dict[pos_node_map[pos]].append(pos_node_map[connection_pos])
                except:
                    print(f"{connection_pos} \n"
                          f"{pos_node_map}")
                    print(f"connection pos: {type(connection_pos)}")
                    print(f"first pos_node_map pos: {type(pos_node_map.keys()[0])}")
                    raise
        return graph_dict


    def toggle_allow_diagonal_connections(self, disabler):
        self._allow_diagonal_connections = not self._allow_diagonal_connections

        print(f"{len(self._diagonal_connections)}\n {self._diagonal_connections}")
        import time
        print("start")
        tic = time.perf_counter()
        if self._allow_diagonal_connections:
            print("enable")
            self.enable_edges(self._diagonal_connections, disabler)
        else:
            print("disable")
            self.disable_edges(self._diagonal_connections, disabler)

        toc = time.perf_counter()
        print(f"Toggled the diagonal connections in {toc - tic:0.4f} seconds")

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

# class ToggleableGridSystem(GridSystem):
#     def __init__(self,
#                  nRows,
#                  nColumns,
#                  toggles: Dict[str, BooleanToggleable],
#                  neighbor_handler: Callable[[GridState, GridState], GridState] = None):
#         GridSystem.__init__(self, nRows, nColumns, default_state=toggles, grid_select_policies=[])
#         self.neighbor_handler = neighbor_handler
#
#     def toggle_at_rc(self, row, column):
#         if row < self.nRows and column < self.nColumns:
#             new_state = self.grid[row][column].toggle()
#             self.handle_others(new_state)
#
#     def handle_others(self, new_state):
#         if self.neighbor_handler is None:
#             return
#
#         for other_grid in self.grid_enumerator():
#             if other_grid.row != new_state.row and other_grid.column != new_state.column:
#                 self.neighbor_handler(new_state, other_grid)

