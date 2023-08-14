from cooptools.sectors.grids import Grid
from coopgraph.graphs import Graph, Node, AStarResults
from coopstructs.toggles import BooleanToggleable
from typing import List, Dict, Tuple
from cooptools.dictPolicies import IActOnDictPolicy

class GridGraph:

    def __init__(self,
                 grid: Grid,
                 connect_adjacents: bool = True,
                 connect_diagonals: bool = True,
                 ):

        self.toggle_key = 'toggle'

        self._diagonal_connections = {}
        self._connect_adjacents = connect_adjacents
        self._connect_diagonals = connect_diagonals
        self.enable_diagonal_connections = BooleanToggleable(default=True)

        self.grid = grid
        self.pos_node_map = self._build_position_node_map(ncols=self.grid.nColumns, nrows=self.grid.nRows)
        graph_dict = self.build_graph_dict(self.pos_node_map)
        self.graph = Graph(graph_dict=graph_dict)


    def _build_position_node_map(self, ncols: int, nrows: int):
        pos_node_map = {}
        for col in range(0, ncols):
            for row in range(0, nrows):
                pos = (col, row)
                pos_node_map[pos] = Node(str(pos), pos)

        return pos_node_map

    def build_graph_dict(self,
                         pos_node_map: Dict[Tuple[float, ...], Node],
                         connect_adjacents: bool=True,
                         connect_diagonals: bool = True)-> Dict[Node, List[Node]]:
        graph_dict = {}

        for pos in pos_node_map.keys():
            graph_dict[pos_node_map[pos]] = []
            adjacents = [
                (pos[0] - 1, pos[1]) if pos_node_map.get((pos[0] - 1, pos[1]), None) else None,  # left
                (pos[0] + 1, pos[1]) if pos_node_map.get((pos[0] + 1, pos[1]), None) else None,  # right
                (pos[0], pos[1] - 1) if pos_node_map.get((pos[0], pos[1] - 1), None) else None,  # up
                (pos[0], pos[1] + 1) if pos_node_map.get((pos[0], pos[1] + 1), None) else None,  # down
            ]

            diagonals = [
                (pos[0] - 1, pos[1] - 1) if pos_node_map.get((pos[0] - 1, pos[1] - 1), None) else None, # UpLeft
                (pos[0] + 1, pos[1] - 1) if pos_node_map.get((pos[0] + 1, pos[1] - 1), None) else None, # UpRight
                (pos[0] - 1, pos[1] + 1) if pos_node_map.get((pos[0] - 1, pos[1] + 1), None) else None, # DownLeft
                (pos[0] + 1, pos[1] + 1) if pos_node_map.get((pos[0] + 1, pos[1] + 1), None) else None # DownRight
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
        self.enable_diagonal_connections.toggle()

        logging.info(f"{'Disabling' if self.enable_diagonal_connections.value is False else 'Enabling'} {len(self._diagonal_connections)} connections")
        import time
        tic = time.perf_counter()
        if self.enable_diagonal_connections.value:
            self.graph.enable_edges(self._diagonal_connections, disabler)
        else:
            self.graph.disable_edges(self._diagonal_connections, disabler)

        toc = time.perf_counter()
        logging.info(f"Toggled the diagonal connections in {toc - tic:0.4f} seconds")

    def astar_between_grid_pos(self, start: Tuple[int, ...], end: Tuple[int, ...]) -> AStarResults:
        results = None
        if start and end:
            start = self.graph.nodes_at_point(start)[0]
            end = self.graph.nodes_at_point(end)[0]
            results = self.graph.astar(start, end)
        return results

    def act_on_grid(self, row: int, column: int, policies:List[IActOnDictPolicy]):
        ret = self.grid.act_at_loc(row, column, policies)

        if self.grid.at(row, column)[self.toggle_key].value:
            self.graph.disable_edges_to_node(self.graph.nodes_at_point((column, row))[0], self.toggle_key)
        else:
            self.graph.enable_edges_to_node(self.graph.nodes_at_point((column, row))[0], self.toggle_key)

        return ret

if __name__ == "__main__":
    from cooptools.sectors.grids import RectGrid
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

    start = (0, 0)
    end = (dimension - 1, dimension - 1)

    logging.info(f"Starting Astar")
    tic = time.perf_counter()

    results = grid_graph.astar_between_grid_pos(start, end)

    [print(x) for x in results.path]
    toc = time.perf_counter()
    logging.info(f"Done {toc - tic}")