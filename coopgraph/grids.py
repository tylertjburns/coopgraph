from coopgraph.graphs import Node
from typing import Dict, List
from coopstructs.vectors import Vector2
from coopstructs.geometry import Rectangle
from coopgraph.dataStructs import GridPoint, UnitPathDefinition
from coopgraph.AGrid import AGrid, IOnGridSelectPolicy


class GridSystem(AGrid):
    def __init__(self, nRows: int, nColumns: int, allow_diagonal_connections:bool=True, grid_select_policies: List[IOnGridSelectPolicy] = None):
        self._diagonal_connections = {}
        self._allow_diagonal_connections = allow_diagonal_connections
        AGrid.__init__(self, nRows=nRows, nColumns=nColumns, grid_select_policies=grid_select_policies)


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
                    print(f"first pos_node_map pos: {type([x for x in pos_node_map.keys()][0])}")
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
