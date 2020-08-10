from coopgraph.graphs import Node, Graph
from typing import Dict, List
from coopstructs.vectors import Vector2, IVector
from coopstructs.geometry import Rectangle
from coopgraph.dataStructs import GridPoint, UnitPathDefinition
from coopgraph.IGridSystem import IGridSystem



class GridSystem(IGridSystem):
    def __init__(self, nRows: int, nColumns: int, allow_diagonal_connections:bool=True):
        self._diagonal_connections = {}
        self._allow_diagonal_connections = allow_diagonal_connections
        IGridSystem.__init__(self, nRows=nRows, nColumns=nColumns)


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

    def build_graph_dict(self, pos_node_map: Dict[Vector2, Node], allow_diagonals: bool = True):
        graph_dict = {}

        for pos in pos_node_map.keys():
            graph_dict[pos_node_map[pos]] = []
            connections = [
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

            if allow_diagonals:
                connections += diagonals

            for connection in diagonals:
                if connection:
                    self._diagonal_connections.setdefault(pos, []).append(connection)

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




# class GridSystem(Graph):
#     def __init__(self, nRows: int, nColumns: int, allow_diagonal_connections:bool=True, default_value = None):
#         if default_value is None:
#             default_value = 0
#         self.nRows = nRows
#         self.nColumns = nColumns
#         self.grid =[[default_value for x in range(self.nColumns)] for y in range(self.nRows)]
#         self.pos_node_map = self.build_position_node_map(nCols=nColumns, nRows=nRows)
#         self._diagonal_connections = {}
#         self._allow_diagonal_connections = allow_diagonal_connections
#         graph_dict = self.build_graph_dict(self.pos_node_map, allow_diagonals=allow_diagonal_connections)
#         # print(f"Length: {len(graph_dict)} \n{graph_dict}")
#         Graph.__init__(self, graph_dict)
#
#
#     def grid_unit_width(self, area_rect:Rectangle):
#         return area_rect.width / self.nColumns
#
#     def grid_unit_height(self, area_rect: Rectangle):
#         return area_rect.height / self.nRows
#
#     def coord_from_grid_pos(self, grid_pos: Vector2, area_rect: Rectangle, grid_point: GridPoint = GridPoint.CENTER) -> Vector2:
#         offset = None
#         if grid_point == GridPoint.CENTER:
#             offset = 0.5
#         elif grid_point == GridPoint.ORIGIN:
#             offset = 0
#         else:
#             raise NotImplementedError(f"Unhandled grid point {grid_point}")
#         grid_unit_width = self.grid_unit_width(area_rect)
#         grid_unit_height = self.grid_unit_height(area_rect)
#
#         ret = Vector2(grid_unit_width * (grid_pos.x + offset), grid_unit_height * (grid_pos.y + offset))
#         return ret
#
#     def at(self, x, y):
#         return self.grid[y][x]
#
#     def grid_from_coord(self, coord: Vector2, area_rect: Rectangle) -> Vector2:
#         if coord is None:
#             return None
#
#         # Change the x/y screen coordinates to grid coordinates
#         column = int((coord.x) // (area_rect.width / self.nColumns))
#         row = int((coord.y) // (area_rect.height / self.nRows))
#
#         grid_coord = Vector2(column, row)
#         return grid_coord
#
#
#
#     def __str__(self):
#         ret = f"<{self.nRows} x {self.nColumns}> of type {type(self.grid[0][0])}"
#         for x in range(0, self.nRows):
#             ret = ret + f"\n{self.grid[x]}"
#
#         return ret
#
#     def build_graph_dict(self, pos_node_map: Dict[Vector2, Node], allow_diagonals:bool= True):
#         graph_dict = {}
#
#         for pos in pos_node_map.keys():
#             graph_dict[pos_node_map[pos]] = []
#             connections = [
#                 Vector2(pos.x-1, pos.y) if pos_node_map.get(Vector2(pos.x-1, pos.y), None) else None,  # left
#                 Vector2(pos.x+1, pos.y) if pos_node_map.get(Vector2(pos.x+1, pos.y), None) else None,  # right
#                 Vector2(pos.x, pos.y - 1) if pos_node_map.get(Vector2(pos.x, pos.y - 1), None) else None,  # up
#                 Vector2(pos.x, pos.y + 1) if pos_node_map.get(Vector2(pos.x, pos.y + 1), None) else None,  # down
#             ]
#
#             diagonals = [
#                     Vector2(pos.x - 1, pos.y - 1) if pos_node_map.get(Vector2(pos.x - 1, pos.y - 1), None) else None,  # UpLeft
#                     Vector2(pos.x + 1, pos.y - 1) if pos_node_map.get(Vector2(pos.x + 1, pos.y - 1), None) else None,  # UpRight
#                     Vector2(pos.x - 1, pos.y + 1) if pos_node_map.get(Vector2(pos.x - 1, pos.y + 1), None) else None,  # DownLeft
#                     Vector2(pos.x + 1, pos.y + 1) if pos_node_map.get(Vector2(pos.x + 1, pos.y + 1), None) else None  # DownRight
#                 ]
#
#
#
#             if allow_diagonals:
#                 connections += diagonals
#
#             for connection in diagonals:
#                 if connection:
#                     self._diagonal_connections.setdefault(pos, []).append(connection)
#
#             for connection_pos in connections:
#                 try:
#                     if connection_pos:
#                         graph_dict[pos_node_map[pos]].append(pos_node_map[connection_pos])
#                 except:
#                     print(f"{connection_pos} \n"
#                           f"{pos_node_map}")
#                     print (f"connection pos: {type(connection_pos)}")
#                     print(f"first pos_node_map pos: {type(pos_node_map.keys()[0])}")
#                     raise
#         return graph_dict
#
#     def path_between_points(self, point1: Vector2, point2: Vector2):
#
#         if point1 not in self.pos_node_map or point2 not in self.pos_node_map:
#             return None
#
#         astar_ret = self.astar(self.pos_node_map[point1], self.pos_node_map[point2])
#         if astar_ret is None:
#             return None
#
#         return [x.pos for x in astar_ret]
#
#
#     def build_position_node_map(self, nCols: int, nRows: int):
#         pos_node_map = {}
#         for x in range(0, nCols):
#             for y in range(0, nRows):
#                 pos = Vector2(x, y)
#                 pos_node_map[pos] = Node(str(pos), pos)
#
#         return pos_node_map
#
#     def toggle_allow_diagonal_connections(self, disabler):
#         self._allow_diagonal_connections = not self._allow_diagonal_connections
#
#         print(f"{len(self._diagonal_connections)}\n {self._diagonal_connections}")
#         import time
#         print("start")
#         tic = time.perf_counter()
#         if self._allow_diagonal_connections:
#             print("enable")
#             self.enable_edges(self._diagonal_connections, disabler)
#         else:
#             print("disable")
#             self.disable_edges(self._diagonal_connections, disabler)
#
#         toc = time.perf_counter()
#         print(f"Toggled the diagonal connections in {toc - tic:0.4f} seconds")

class ToggleableGridSystem(GridSystem):
    def __init__(self, nRows, nColumns, default: bool = False, preset_grids: List[Vector2] = None):
        GridSystem.__init__(self, nRows, nColumns, default_value=default)
        self._set_presets(preset_grids, "presets")

    def toggle_at_rc(self, row, column, disabler):
        if column < self.nColumns and row < self.nRows:
            self.grid[row][column] = not self.grid[row][column]
            if self.grid[row][column]:
                self.enable_edges_to_node(Vector2(column, row), disabler)
            else:
                self.disable_edges_to_node(Vector2(column, row), disabler)

    def toggle_at_xy(self, x, y, disabler):
        self.toggle_at_rc(y, x, disabler)

    def _set_presets(self, presets: List[Vector2], disabler):
        if presets is None:
            return

        for grid_pos in presets:
            self.toggle_at_xy(grid_pos.x, grid_pos.y, disabler)

    def on(self):
        ret = []
        for x in range(0, self.nColumns):
            for y in range(0, self.nRows):
                if self.grid[y][x]:
                    ret.append(Vector2(x, y))
        return ret

    def off(self):
        ret = []
        for x in range(0, self.nColumns):
            for y in range(0, self.nRows):
                if not self.grid[y][x]:
                    ret.append(Vector2(x, y))
        return ret

class UnitPathGridSystem(GridSystem):
    def __init__(self, nRows, nColumns):
        super().__init__(nRows, nColumns)
        self.grid = [[UnitPathDefinition() for x in range(self.nColumns)] for y in range(self.nRows)]

    def toggle_at_rc(self, row, column):
        if row < self.nRows and column < self.nColumns:
            new_toggle = 1 - self.grid[row][column].toggled
            self.grid[row][column].toggled = new_toggle
            self.handle_neigbors(column, row, new_toggle)

    def toggle_at_xy(self, x, y):
        if y < self.nRows and x < self.nColumns:
            new_toggle = 1 - self.grid[y][x].toggled
            self.grid[y][x].toggled = new_toggle
            self.handle_neigbors(x, y, new_toggle)

    def handle_neigbors(self, x, y, new_toggle):
        # Handle Left
        if x > 0 and self.grid[y][x - 1].toggled:
            self.grid[y][x - 1].right = new_toggle
            self.grid[y][x].left = new_toggle

        # Handle Up
        if y > 0 and self.grid[y - 1][x].toggled:
            self.grid[y - 1][x].down = new_toggle
            self.grid[y][x].up = new_toggle

        # Handle Right
        if x < len(self.grid[y]) - 1 and self.grid[y][x + 1].toggled:
            self.grid[y][x + 1].left = new_toggle
            self.grid[y][x].right = new_toggle

        # Handle Down
        if y < len(self.grid) - 1 and self.grid[y + 1][x].toggled:
            self.grid[y + 1][x].up = new_toggle
            self.grid[y][x].down = new_toggle


class GridState():
    def __init__(self):
        self.state = {}


class StateKeeperGridSystem(GridSystem):
    def __init__(self, nRows, nColumns, preset_grids: Dict[Vector2, GridState] = None):
        GridSystem.__init__(self, nRows, nColumns, default_value=GridState)
        self._set_presets(preset_grids)

    def add_disabler_at_rc(self, row, column, disabler):
        if column < self.nColumns and row < self.nRows:
            self.grid[row][column] = not self.grid[row][column]
            if self.grid[row][column]:
                self.enable_edges_to_node(Vector2(column, row), disabler)
            else:
                self.disable_edges_to_node(Vector2(column, row), disabler)

    def add_disabler_at_xy(self, x: int, y: int, disabler):
        self.add_disabler_at_rc(y, x, disabler)

    def set_state_at_rc(self, row: int, column: int, state: GridState):
        if 0 <= column < self.nColumns and 0 <= row < self.nRows:
            self.grid[row][column] = state

    def set_state_at_xy(self, x: int, y: int, state: GridState):
        self.set_state_at_rc(y, x, state)

    def _set_presets(self, presets: Dict[Vector2, GridState]):
        if presets is None:
            return

        for grid_pos, state in presets.items():
            self.set_state_at_xy(grid_pos.x, grid_pos.y, state=state)

if __name__ == "__main__":

    grid1 = GridSystem(5, 5)
    grid2 = GridSystem(10, 10)

    print(grid1._diagonal_connections)
    print(grid2._diagonal_connections)