from coopgraph.grids import AGrid
from coopgraph.graphs import Graph, Node, AStarResults
from coopstructs.vectors import Vector2, IVector
from typing import List, Dict

class GridGraph:

    def __init__(self,
                 grid: AGrid,
                 connect_adjacents: bool = True,
                 connect_diagonals: bool = True,
                 ):
        self._diagonal_connections = {}
        self._connect_adjacents = connect_adjacents
        self._connect_diagonals = connect_diagonals

        self.grid = grid
        self.pos_node_map = self._build_position_node_map(ncols=self.grid.nColumns, nrows=self.grid.nRows)
        graph_dict = self.graph_dict_provider(self.pos_node_map)
        self.graph = Graph(graph_dict=graph_dict)


    def _build_position_node_map(self, ncols: int, nrows: int):
        pos_node_map = {}
        for col in range(0, ncols):
            for row in range(0, nrows):
                pos = Vector2(col, row)
                pos_node_map[pos] = Node(str(pos), pos)

        return pos_node_map

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
            self.graph.enable_edges(self._diagonal_connections, disabler)
        else:
            print("disable")
            self.graph.disable_edges(self._diagonal_connections, disabler)

        toc = time.perf_counter()
        print(f"Toggled the diagonal connections in {toc - tic:0.4f} seconds")

    def astar_between_grid_pos(self, start: Vector2, end: Vector2) -> AStarResults:
        start = self.graph.nodes_at_point(start)[0]
        end = self.graph.nodes_at_point(end)[0]
        results = self.graph.astar(start, end)
        return results

if __name__ == "__main__":
    from coopgraph.grids import RectGrid
    import logging
    import time

    logging.basicConfig(level=logging.INFO)

    dimension = 100

    logging.info(f"Creating Grid")
    tic = time.perf_counter()
    grid = RectGrid(dimension, dimension)
    toc = time.perf_counter()
    logging.info(f"Done {toc - tic}")

    logging.info(f"Creating Graph")
    tic = time.perf_counter()
    grid_graph = GridGraph(grid=grid, connect_diagonals=False)
    toc = time.perf_counter()
    logging.info(f"Done {toc - tic}")

    start = Vector2(0, 0)
    end = Vector2(dimension - 1, dimension - 1)

    logging.info(f"Starting Astar")
    tic = time.perf_counter()

    results = grid_graph.astar_between_grid_pos(start, end)

    [print(x) for x in results.path]
    toc = time.perf_counter()
    logging.info(f"Done {toc - tic}")