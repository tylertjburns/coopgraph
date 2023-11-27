""" A Python Class
A simple Python graph class, demonstrating the essential
facts and functionalities of graphs.
"""
from enum import Enum
from typing import List, Dict, Callable, Tuple
import uuid
import logging
import copy
import cooptools.geometry_utils.vector_utils as vec_util
from cooptools.common import flattened_list_of_lists

class Node(object):
    def __init__(self, name:str, pos: Tuple[float, ...]):
        if not isinstance(pos, Tuple) :
            raise TypeError(f"position must be of type {Tuple[float, ...]}, but {type(pos)} was provided")

        self.name = name
        self.pos = pos

    def __str__(self):
        return f"{str(self.name)} at {self.pos}"

    def __eq__(self, other):
        if isinstance(other, Node) and other.name == self.name:
            return True
        else:
            return False

    def __hash__(self):
        return hash(str(self.name))

    def __repr__(self):
        return self.__str__()

class _AStarMetrics():
    def __init__(self, parent, graph_node: Node):
        if not (isinstance(parent, _AStarMetrics) or parent is None):
            raise TypeError(f"Astar parent must be AStarNode or None, {type(parent)} was given")

        if not (isinstance(graph_node, Node)):
            raise TypeError(f"Astar graph_node must be Node, {type(graph_node)} was given")

        self.parent = parent
        self.graph_node = graph_node
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        if isinstance(other, _AStarMetrics) and other.graph_node == self.graph_node:
            return True
        else:
            return False

    def __hash__(self):
        return self.graph_node.__hash__()

    def __repr__(self):
        return f"{self.graph_node} g: {self.g} h: {self.h} f: {self.f}"


class AStarResults:
    def __init__(self, path, steps):
        self.path = path
        self.steps = steps




class Edge(object):
    def __init__(self,
                 nodeA: Node,
                 nodeB: Node,
                 edge_weight: float = None,
                 naming_provider: Callable[[], str] = None):
        self.start = nodeA
        self.end = nodeB
        self._disablers = set()
        self.length = vec_util.distance_between(nodeA.pos, nodeB.pos)
        self.id = naming_provider() if naming_provider else str(uuid.uuid4())
        self.weight = edge_weight

    def __str__(self):
        return f"{self.start.name}-->{self.end.name}"

    def __hash__(self):
        return hash(str(self.id))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Edge) and other.id == self.id:
            return True
        else:
            return False

    def matches_profile_of(self, other):
        if isinstance(other, Edge) and other.start == self.start and self.end == other.end:
            return True
        else:
            return False

    def eucledian_distance(self):
        return self.length

    def enabled(self, ignored_disablers: set = None):
        if ignored_disablers is None:
            ignored_disablers = set()
        return self._disablers.issubset(ignored_disablers)


    def remove_disabler(self, disabler):
        self._disablers.discard(disabler)

    def add_disabler(self, disabler):
        self._disablers.add(disabler)

    def config_match(self, other):
        if isinstance(other, Edge) and other.start == self.start and other.end == self.end and other._disablers == self._disablers:
            return True
        else:
            return False

    def disablers(self):
        return copy.deepcopy(self._disablers)

class Graph(object):

    #TODO: Add a "from file" constructor cls method

    # def create_points_from_file(filePath: str, scale: (int, int), hasHeaders: bool = True):
    #     import csv
    #
    #     points = []
    #     with open(filePath) as csv_file:
    #         csv_reader = csv.reader(csv_file, delimiter=',')
    #         line_count = 0
    #
    #         for row in csv_reader:
    #             if line_count == 0 and hasHeaders:
    #                 print(f'Column names are {", ".join(row)}')
    #                 line_count += 1
    #             else:
    #                 # end_node= Node(str(row[0]) + "_start", int(float(row[1])), int(float(row[3]))
    #
    #                 points.append((int(float(row[1]) * scale[0]), int(float(row[3]) * scale[1])))
    #                 line_count += 1
    #
    #     return points
    #
    # def create_edges_from_file(filePath: str, scale: (int, int), hasHeaders: bool = True):
    #     import csv
    #
    #     edges = []
    #     with open(filePath) as csv_file:
    #         csv_reader = csv.reader(csv_file, delimiter=',')
    #         line_count = 0
    #
    #         for row in csv_reader:
    #             if line_count == 0 and hasHeaders:
    #                 print(f'Column names are {", ".join(row)}')
    #                 line_count += 1
    #             else:
    #                 start_node = Node(str(row[0]) + "_start", int(float(row[1]) * scale[0]),
    #                                   int(float(row[2]) * scale[1]))
    #                 end_node = Node(str(row[0]) + "_end", int(float(row[3]) * scale[0]), int(float(row[4]) * scale[1]))
    #                 edges.append(Edge(row[0], start_node, end_node, EdgeConnection.TwoWay))
    #                 line_count += 1
    #
    #     return edges



    def __init__(self, graph_dict: Dict[Node, List[Node]]=None, naming_provider: Callable[[], str] = None):
        """ initializes a graph object
            If no dictionary or None is given, an empty dictionary will be used
        """
        if graph_dict is None:
            graph_dict = {}
        self._graph_dict = graph_dict

        self.naming_provider = naming_provider

        self._nodes_dict = {}
        self._edges_dict = {}

        ''' self._edges[start: Vector2][end: Vector2] = edge: Edge '''
        self._pos_edge_map = None
        ''' dict with key: Node, value: edge_names: List[str] '''
        self._node_edge_map = None
        ''' dict with key: IVector, value: node_id '''
        self._pos_node_map = None
        ''' dict with key: node_name, value node_id'''
        self._node_by_name_map = None


        for node in graph_dict.keys():
            self._nodes_dict[node.name] = node
            for toNode in graph_dict[node]:
                edge = Edge(node, toNode, naming_provider=self.naming_provider)
                self._edges_dict[edge.id] = edge

        self._build_maps()


    def _build_maps(self):
        # self._graph_dict is a first class citizen. Therefore update all the others off of an updated graph_dict
        self._graph_dict = self.__generate_graph_dict(self._edges_dict)

        # update remaining maps assuming graph_dict is accurate
        self._pos_edge_map = self.__generate_pos_edge_map(self._edges_dict)
        self._node_edge_map = self.__generate_node_edge_map(self._edges_dict)
        self._pos_node_map = self.__generate_position_node_map(self._nodes_dict)
        self._node_by_name_map = self.__generate_node_by_name_map(self._nodes_dict)
        self._node_to_node_edge_map = self.__generate_node_to_node_edge_map(self._edges_dict)



    def __generate_node_edge_map(self, edges: Dict[str, Edge]):
        node_edge_map = {}

        for node in self._graph_dict.keys():
            node_edge_map[node] = []

        for id, edge in edges.items():
            node_edge_map.setdefault(edge.start, []).append(edge.id)
            node_edge_map.setdefault(edge.end, []).append(edge.id)
        return node_edge_map

    def __generate_pos_edge_map(self, edges):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the node) or two
            vertices
        """
        edges_by_pos_dict = {}

        for id, edge in edges.items():
            edges_by_pos_dict.setdefault(edge.start.pos, {}).setdefault(edge.end.pos, []).append(id)

        return edges_by_pos_dict

    def __generate_node_to_node_edge_map(self, edges):
        edge_ids_by_node = {}

        for id, edge in edges.items():
            edge_ids_by_node.setdefault(edge.start, {})[edge.end] = id

        return edge_ids_by_node

    def __generate_position_node_map(self, nodes):
        pos_node_map = {}
        for id, node in nodes.items():
            pos = node.pos
            pos_node_map.setdefault(pos, []).append(id)

        return pos_node_map


    def __generate_graph_dict(self, edges):
        graph_dict = {}
        for id, edge in edges.items():
            graph_dict.setdefault(edge.start, []).append(edge.end)

        for id, node in self._nodes_dict.items():
            if node not in graph_dict.keys():
                graph_dict[node] = []
        return graph_dict

    def __generate_node_by_name_map(self, nodes):
        ret = {}
        for node_id, node in nodes.items():
            ret[node.name] = node_id
        return ret

    def _nodes(self) -> List[Node]:
        """ returns the vertices of a graph """
        # return copy.deepcopy([node for id, node in self._nodes.items()])
        return ([node for id, node in self._nodes_dict.items()])

    @property
    def nodes(self) -> List[Node]:
        return self._nodes()

    def _edges(self, ids: List[str] = None) -> List[Edge]:
        """ returns the edges of a graph """
        # return copy.deepcopy([edge for id, edge in self._edges.items()])
        return ([edge for id, edge in self._edges_dict.items() if ids is None or id in ids])

    @property
    def edges(self) -> List[Edge]:
        return self._edges()

    def edges_by_id(self, edge_ids: List[str]):
        return self._edges(edge_ids)

    @property
    def DestinationNodes(self):
        return flattened_list_of_lists(self._graph_dict.values(), unique=True)

    @property
    def Sources(self) -> List[Node]:
        dests = self.DestinationNodes
        return [n for n, cnxn in self._graph_dict.items() if n not in dests and len(cnxn) != 0]

    @property
    def Sinks(self) -> List[Node]:
        return [n for n, cnxn in self._graph_dict.items() if len(cnxn) == 0 and n not in self.Orphans]

    @property
    def Orphans(self) -> List[Node]:
        return [n for n, edges in self._node_edge_map.items() if len(edges) == 0]

    def add_node_with_connnections(self, node: Node, connections: List[Node]):
        self.add_node(node)
        edges = []

        for connection in connections:
            edges.append(Edge(node, connection, naming_provider=self.naming_provider))

        self.add_edges(edges)

    def add_node(self, node):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if node.name not in self._nodes_dict.keys():
            self._nodes_dict[node.name] = node
        self._build_maps()

    def add_edges(self, edges: List[Edge]):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        if isinstance(edges, list) and len(edges) > 0 and isinstance(edges[0], Edge):
            for edge in edges:
                self._add_edge(edge)
        elif isinstance(edges, dict) and isinstance(list(edges.keys())[0], Tuple):
            for start in edges.keys():
                for end in edges[start]:
                    start_node = self.nodes_at_point(start)[0]
                    end_node = self.nodes_at_point(end)[0]
                    if start_node and end_node:
                        edge = Edge(start_node, end_node, naming_provider=self.naming_provider)
                        self._add_edge(edge)
        elif isinstance(edges, Edge):
            self._add_edge(edges)
        else:
            raise ValueError()

    def remove_edges(self, edges):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        print(f"len start {len(self._pos_edge_map)}")
        if isinstance(edges, list) and isinstance(edges[0], Edge):
            for edge in edges:
                self._remove_edge(edge)
        elif isinstance(edges, dict) and isinstance(list(edges.keys())[0], Tuple):
            for start in edges.keys():
                for end in edges[start]:
                    start_node = self.nodes_at_point(start)[0]
                    end_node = self.nodes_at_point(end)[0]
                    if start_node and end_node:
                        edge = self._edge_at(start_node.pos, end_node.pos)
                        self._remove_edge(edge)
        elif isinstance(edges, Edge):
            self._remove_edge(edges)

        print(f"len end {len(self._pos_edge_map)}")

    def enable_edges(self, edges, disabler):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        if isinstance(edges, list) and isinstance(edges[0], Edge):
            for edge in edges:
                edge.remove_disabler(disabler)
        elif isinstance(edges, dict) and isinstance(list(edges.keys())[0], Tuple):
            for start in edges.keys():
                for end in edges[start]:
                    edge = self._edge_at(start, end)
                    if edge:
                        edge.remove_disabler(disabler)
        elif isinstance(edges, Edge):
            edges.remove_disabler(disabler)

    def disable_edges(self, edges, disabler):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        if isinstance(edges, list) and isinstance(edges[0], Edge):
            for edge in edges:
                edge.add_disabler(disabler)
        elif isinstance(edges, dict) and isinstance(list(edges.keys())[0], Tuple):
            for start in edges.keys():
                for end in edges[start]:
                    edge = self._edge_at(start, end)
                    if edge:
                        edge.add_disabler(disabler)
        elif isinstance(edges, Edge):
            edges.add_disabler(disabler)

    def edges_to_node(self, node: Node, only_enabled: bool = False, ignored_disablers: List[str] = None):
        if not isinstance(node, Node):
            raise Exception(f"input must be of type {Node}, but {type(node)} was provided")
        node_edges = self._node_edge_map[node]
        ret = [self._edges_dict[edge_id] for edge_id in node_edges if self._edges_dict[edge_id].end == node]

        if only_enabled:
            ret = [x for x in ret if x.enabled(ignored_disablers)]

        return ret

    def edges_including_node(self, node: Node, only_enabled: bool = False, ignored_disablers: List[str] = None):
        if not isinstance(node, Node):
            raise Exception(f"input must be of type {Node}, but {type(node)} was provided")
        node_edges = self._node_edge_map[node]
        ret = [self._edges_dict[edge_id] for edge_id in node_edges]

        if only_enabled:
            ret = [x for x in ret if x.enabled(ignored_disablers)]

        return ret

    def edges_from_node(self, node: Node, only_enabled: bool = False, ignored_disablers: List[str] = None):
        if not isinstance(node, Node):
            raise Exception(f"input must be of type {Node}, but {type(node)} was provided")
        node_edges = self._node_edge_map[node]
        ret = [self._edges_dict[edge_id] for edge_id in node_edges if self._edges_dict[edge_id].start == node]

        if only_enabled:
            ret = [x for x in ret if x.enabled(ignored_disablers)]

        return ret


    def disable_edges_to_node(self, node: Node, disabler):
        logging.debug(f"disable {node} with disabler {disabler}")

        if isinstance(node, Node):
            node_edges = self._node_edge_map[node]
            for edge_id in node_edges:
                edge = self._edges_dict[edge_id]
                edge.add_disabler(disabler)

    def enable_edges_to_node(self, node: Node, disabler):
        logging.debug(f"enable {node} on disabler {disabler}")

        if isinstance(node, Node):
            node_edges = self._node_edge_map[node]
            for edge_id in node_edges:
                edge = self._edges_dict[edge_id]
                edge.remove_disabler(disabler)

    def adjacent_nodes(self, node: Node, only_enabled: bool = False, ignored_disablers: set = None) -> List[Node]:
        adjacents = list(self._graph_dict[node])
        if only_enabled:
            adjacents[:] = [x for x in adjacents if self._edge_at(node.pos, x.pos).enabled(ignored_disablers)]

        return adjacents

    def _remove_edge(self, edge: Edge):
        edge = self._edge_at(edge.start.pos, edge.end.pos)
        if edge.id in self._pos_edge_map.keys():
            del self._edges_dict[edge.id]
        self._build_maps()

    def _add_edge(self, edge: Edge):
        if edge.id not in self._edges_dict.keys():
            self._edges_dict[edge.id] = edge
        else:
            raise ValueError(f"Edge with id: {edge.id} already exists")
        self._build_maps()

    def nodes_at_point(self, pos: Tuple[float, ...]) -> List[Node]:
        node_ids = self._pos_node_map.get(pos, [])

        return [self._nodes_dict.get(node_id, None) for node_id in node_ids]

    def nodes_at(self, points: List[Tuple[float, ...]]) -> Dict[Tuple[float, ...], List[Node]]:
        return {point: self.nodes_at_point(point) for point in points}

    def _edge_at(self, start: Tuple[float, ...], end: Tuple[float, ...]):
        # start = self.nodes_at_point(start).pos
        # end = self.nodes_at_point(end).pos

        if not (start and end):
            return None

        #TODO: Naively taking first entry between positions. Need to be more explicit on implemetnation since there could be multiple edges
        edges = self._pos_edge_map.get(start, {}).get(end, None)
        edge_id = edges[0] if edges is not None else None
        if edge_id:
            return self._edges_dict.get(edge_id, None)

        return None


    def edge_between(self, nodeA: Node, nodeB: Node):
        try:
            edge_id = self._node_to_node_edge_map[nodeA][nodeB]
            return self._edges_dict.get(edge_id, None)
        except:
            return None


    def __str__(self):
        res = "vertices: "
        for id, node in self._nodes_dict.items():
            res += f"\n\t{str(node)}"
        res += "\nedges: "
        for edge in self.edges:
            res += f"\n\t{str(edge)}"
        return res

    def find_isolated_vertices(self):
        """ returns a list of isolated vertices. """
        isolated = []
        for node in self._node_edge_map.keys():
            if len(self._node_edge_map[node]) == 0:
                isolated += [node]
        return isolated

    def find_path(self, start_vertex: Node, end_vertex: Node, path=None):
        """ find a path from start_vertex to end_vertex
            in graph """
        if path is None:
            path = []
        nodes = self._nodes_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return path
        if start_vertex.name not in nodes.keys() or end_vertex.name not in nodes.keys():
            return None
        for node, edge_id in self._node_edge_map.items():
            if node not in path:
                extended_path = self.find_path(node,
                                               end_vertex,
                                               path)
                if extended_path:
                    return extended_path
        return None

    def find_all_paths(self, start_vertex: Node, end_vertex: Node, path=None):
        """ find all paths from start_vertex to
            end_vertex in graph """
        if path is None:
            path = []
        nodes = self._nodes_dict
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex.name not in nodes.keys() or end_vertex.name not in nodes.keys():
            return []
        paths = []
        for edge_id in self._node_edge_map[start_vertex]:
            edge = self._edges_dict[edge_id]
            node = edge.end
            if node not in path:
                extended_paths = self.find_all_paths(node,
                                                     end_vertex,
                                                     path)
                for p in extended_paths:
                    paths.append(p)
        return paths

    def vertex_degree(self, node: Node):
        """ The degree of a vertex is the number of edges connecting
            it, i.e. the number of adjacent vertices. Loops are counted
            double, i.e. every occurence of vertex in the list
            of adjacent vertices. """

        edges = self._node_edge_map.get(node, [])
        edges = [self._edges_dict[x] for x in edges]
        degree = len(edges) + edges.count([x for x in edges if x.end == node])
        return degree


    def degree_sequence(self):
        """ calculates the degree sequence """
        seq = []
        for id, node in self._nodes_dict.items():
            seq.append(self.vertex_degree(node))
        seq.sort(reverse=True)
        return tuple(seq)


    @staticmethod
    def is_degree_sequence(sequence):
        """ Method returns True, if the sequence "sequence" is a
            degree sequence, i.e. a non-increasing sequence.
            Otherwise False is returned.
        """
        # check if the sequence sequence is non-increasing:
        return all(x >= y for x, y in zip(sequence, sequence[1:]))


    def delta(self):
        """ the minimum degree of the vertices """
        min = 100000000
        for id, node in self._nodes_dict.items():
            vertex_degree = self.vertex_degree(node)
            if vertex_degree < min:
                min = vertex_degree
        return min


    def Delta(self):
        """ the maximum degree of the vertices """
        max = 0
        for id, node in self._nodes_dict.items():
            vertex_degree = self.vertex_degree(node)
            if vertex_degree > max:
                max = vertex_degree
        return max


    def density(self):
        """ method to calculate the density of a graph """
        g = self._nodes_dict
        V = len(g.keys())
        E = len(self.edges())
        return 2.0 * E / (V * (V - 1))


    def diameter(self):
        """ calculates the diameter of the graph """

        v = self.nodes()
        pairs = [(v[i], v[j]) for i in range(len(v)) for j in range(i + 1, len(v) - 1)]
        smallest_paths = []
        for (s, e) in pairs:
            paths = self.find_all_paths(s, e)
            smallest = sorted(paths, key=len)[0]
            smallest_paths.append(smallest)

        smallest_paths.sort(key=len)

        # longest path is at the end of list,
        # i.e. diameter corresponds to the length of this path
        diameter = len(smallest_paths[-1]) - 1
        return diameter


    @staticmethod
    def erdoes_gallai(dsequence):
        """ Checks if the condition of the Erdoes-Gallai inequality
            is fullfilled
        """
        if sum(dsequence) % 2:
            # sum of sequence is odd
            return False
        if Graph.is_degree_sequence(dsequence):
            for k in range(1, len(dsequence) + 1):
                left = sum(dsequence[:k])
                right = k * (k - 1) + sum([min(x, k) for x in dsequence[k:]])
                if left > right:
                    return False
        else:
            # sequence is increasing
            return False
        return True

    def astar(self,
              start: Node,
              end: Node,
              g_func: Callable[[Node, Node], float] = None,
              h_func: Callable[[Node, Node], float] = None,
              ignored_disablers:List[str]=None) -> AStarResults:
        if not ignored_disablers:
            ignored_disablers = []

        """Returns a list of nodes as a path from the given start to the given end in the given graph"""
        logging.debug(f"Performing A* over map of length: {len(self._graph_dict)}")

        # Create start and end node
        start_iter = _AStarMetrics(None, start)
        end_iter = _AStarMetrics(None, end)

        steps = {}

        # Initialize both open and closed list
        open_set = set()
        closed_set = set()

        enabled_connections_to_end = self.edges_to_node(end, only_enabled=True, ignored_disablers=ignored_disablers)
        if len(enabled_connections_to_end) != 0:
            # Add the start node
            open_set.add(start_iter)

        cc = -1


        results = None
        # Loop until you find the end
        while len(open_set) > 0:
            cc += 1

            # Find the node on open list with the least F value
            current_item = next(iter(open_set))
            for open_item in open_set:
                if open_item.f < current_item.f:
                    current_item = open_item

            steps[cc] = {"open_set": set(open_set), "closed_set": set(closed_set), "current_item": current_item}

            # Pop current off open list, add to closed list
            open_set.remove(current_item)
            closed_set.add(current_item)

            # Found the goal
            if current_item == end_iter:
                path = []
                current = current_item
                while current is not None:
                    path.append(current.graph_node)
                    current = current.parent

                results = AStarResults(path[::-1], steps)  # Return reversed path
                break

            # Generate children
            for new_node in self._graph_dict[current_item.graph_node]:  # Adjacent nodes
                # Dont evaluate this node if the edge is not enabled
                if not self._edge_at(current_item.graph_node.pos, new_node.pos).enabled(ignored_disablers=set(ignored_disablers)):
                    continue

                new_item = _AStarMetrics(current_item, new_node)

                if new_item in closed_set:
                    continue

                open_set.add(new_item)

                # calculate new g, h, f from current pivot node to the new node
                calc_g = current_item.g + g_func(current_item.graph_node,
                                                 new_node) if g_func else vec_util.distance_between(new_node.pos, current_item.graph_node.pos)
                calc_h = h_func(new_node,
                                end_iter.graph_node) if h_func else vec_util.distance_between(new_node.pos, end_iter.graph_node.pos)
                calc_f = calc_g + calc_h

                new_item.parent = current_item
                new_item.g = calc_g
                new_item.h = calc_h
                new_item.f = calc_f

        if results is None:
            ''' No Path Found '''
            logging.error(f"Unable to find a path from [{start}] to [{end}]")

            return AStarResults(None, steps)
        else:
            ''' Log Path found '''
            logging.debug(f"Path found from [{start}] to [{end}] in {len(steps)} steps")
            return results

    def path_length(self, path:List[Node]):
        length = 0
        last = None
        for node in path:
            if last is not None:
                dist = node.pos.distance_from(last.pos)
                # print(f"{last} to {node} = {dist}")
                length += dist
            last = node

        return length

    def node_by_name(self, node_name: str):
        # nodes = self.nodes()
        # return next(node for node in nodes if node.name == node_name)
        return self._nodes_dict[self._node_by_name_map[node_name]]



    def verify_edge_configuration(self, edges_to_compare: List[Edge]):
        for edge in edges_to_compare:
            if not self._edges_dict[self._pos_edge_map[edge.start.pos][edge.end.pos]].config_match(edge):
                return False

        return True

    def APUtil(self, u, visited, ap, parent, low, disc, iter):
        '''A recursive function that find articulation points
            using DFS traversal
            u --> The vertex to be visited next
            visited[] --> keeps tract of visited vertices
            disc[] --> Stores discovery times of visited vertices
            parent[] --> Stores parent vertices in DFS tree
            ap[] --> Store articulation points'''

        # Count of children in current node
        children = 0

        # Mark the current node as visited and print it
        visited[u] = True

        # Initialize discovery time and low value
        disc[u] = iter
        low[u] = iter
        new_iter = iter + 1

        # Recur for all the vertices adjacent to this vertex
        for v in self._graph_dict[u]:
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if visited[v] == False:
                parent[v] = u
                children += 1
                self.APUtil(v, visited, ap, parent, low, disc, new_iter)

                # Check if the subtree rooted with v has a connection to
                # one of the ancestors of u
                low[u] = min(low[u], low[v])

                # u is an articulation point in following cases
                # (1) u is root of DFS tree and has two or more chilren.
                if parent[u] == -1 and children > 1:
                    ap[u] = children

                # (2) If u is not root and low value of one of its child is more
                # than discovery value of u.
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap[u] = children

                    # Update low value of u for parent function calls
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

                # The function to do DFS traversal. It uses recursive APUtil()

    def AP(self) -> Dict[Node, int]:
        # https://www.geeksforgeeks.org/articulation-points-or-cut-vertices-in-a-graph/

        # Mark all the vertices as not visited
        # and Initialize parent and visited,
        # and ap(articulation point) arrays
        visited = {node: False for id, node in self._nodes_dict.items()}
        disc = {node: float("Inf") for node in self._graph_dict.keys()}
        low = {node: float("Inf") for node in self._graph_dict.keys()}
        parent = {node: -1 for node in self._graph_dict.keys()}
        ap = {node: 0 for node in self._graph_dict.keys()}  # To store articulation points

        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in self._graph_dict.keys():
            if visited[i] == False:
                self.APUtil(i, visited, ap, parent, low, disc, 0)

        ret = {k: v for k, v in ap.items() if v > 0}

        logging.debug(f"Articulation points evaluation:"
                      f"\n\tVisited: {visited}"
                      f"\n\tdisc: {disc}"
                      f"\n\tlow: {low}"
                      f"\n\tparent: {parent}"
                      f"\n\tap: {ap}"
                      f"\nAPs: [{[k for k, v in ret.items()]}]")

        return ret


    def closest_nodes(self, pos: Tuple[float, ...]) -> List[Node]:
        closest_nodes = None
        closest_distance = None
        for node in self.nodes:
            distance = vec_util.distance_between(node.pos, pos)
            if closest_nodes is None or distance < closest_distance:
                closest_nodes = [node]
                closest_distance = distance

            # Add node to return list if found multiple at distance
            if distance == closest_distance:
                closest_nodes.append(node)

        return closest_nodes


    #TODO: enable a "walk" method. should accept a number of steps, an optional start point, and and optional end point

    # def walk(self, name: str, points: [(int, int)]) -> ({str, Node}, {str, Edge}):
    #     walk_nodes: {str, Node} = {}
    #     walk_edges: {str, Edge} = {}
    #
    #     old_point = None
    #     for point in points:
    #         new_point = Node(self.nodeIndex, point[0], point[1])
    #         walk_nodes[str(self.nodeIndex)] = new_point
    #         self.nodeIndex += 1
    #         if old_point is not None:
    #             walk_edges[str(self.edgeIndex)] = Edge(self.edgeIndex, old_point, new_point, EdgeConnection.OneWay)
    #             self.edgeIndex += 1
    #
    #         old_point = new_point
    #
    #     self.nodes.update(walk_nodes)
    #     self.edges.update(walk_edges)
    #     ret = (walk_nodes, walk_edges)
    #     self.walks[name] = ret
    #     return ret

    def copy(self):
        copy = Graph(graph_dict=self._graph_dict)
        for edge in copy.edges:
            og_edge = self.edge_between(edge.start, edge.end)
            for disabler in og_edge.disablers():
                edge.add_disabler(disabler)

        return copy

if __name__ == "__main__":
    pass
