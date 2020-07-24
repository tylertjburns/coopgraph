""" A Python Class
A simple Python graph class, demonstrating the essential
facts and functionalities of graphs.
"""
from coopgraph.dataStructs import IVector
from enum import Enum
from typing import List, Dict, Callable
import uuid
import logging
import copy

class EdgeDirection(Enum):
    ONEWAY = 1
    TWOWAY = 2

class Node(object):
    def __init__(self, name:str, pos: IVector):
        if not isinstance(pos, IVector):
            raise TypeError(f"position must be of type {type(IVector)}, but {type(pos)} was provided")

        self.name = name
        self.pos = pos

    def __str__(self):
        return f"{str(self.name)} at {self.pos}"

    def __eq__(self, other):
        if isinstance(other, Node) and other.name == self.name and self.pos == other.pos:
            return True
        else:
            return False

    def __hash__(self):
        return hash(str(self.name))

    def __repr__(self):
        return self.__str__()

class AStarNode():
    def __init__(self, parent, graph_node: Node):
        if not (isinstance(parent, AStarNode) or parent is None):
            raise TypeError(f"Astar parent must be AStarNode or None, {type(parent)} was given")

        if not (isinstance(graph_node, Node)):
            raise TypeError(f"Astar graph_node must be Node, {type(graph_node)} was given")

        self.parent = parent
        self.graph_node = graph_node
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        if isinstance(other, AStarNode) and other.graph_node == self.graph_node:
            return True
        else:
            return False

    def __hash__(self):
        return self.graph_node.__hash__()

    def __repr__(self):
        return self.graph_node.__str__()

class Edge(object):
    def __init__(self, nodeA: Node, nodeB: Node, edge_direction=None):
        self.start = nodeA
        self.end = nodeB
        if edge_direction is None:
            edge_direction = EdgeDirection.ONEWAY
        self.direction = edge_direction
        self._disablers = set()
        self.length = nodeA.pos.distance_from(nodeB.pos)
        self.id = str(uuid.uuid4())

    def __str__(self):
        char_dict = {
            EdgeDirection.ONEWAY: "->",
            EdgeDirection.TWOWAY: "<->"
        }
        return f"{{{self.start} {char_dict.get(self.direction)} {self.end}}}"

    def __hash__(self):
        return hash(str(self.id))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Edge) and other.start == self.start and self.end == other.end:
            return True
        else:
            return False

    def enabled(self, ignored_disablers: set = None):
        if ignored_disablers is None:
            ignored_disablers = set()
        return True if len(self._disablers - ignored_disablers) == 0 else False


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

    def __init__(self, graph_dict: Dict[Node, List[Node]]=None):
        """ initializes a graph object
            If no dictionary or None is given, an empty dictionary will be used
        """
        if graph_dict is None:
            graph_dict = {}
        self._graph_dict = graph_dict



        self._nodes = {}
        self._edges = {}

        ''' self._edges[start: Vector2][end: Vector2] = edge: Edge '''
        self._pos_edge_map = None
        ''' dict with key: Node, value: edge_names: List[str] '''
        self._node_edge_map = None
        ''' dict with key: IVector, value: node_id '''
        self._pos_node_map = None
        ''' dict with key: node_name, value node_id'''
        self._node_by_name_map = None


        for node in graph_dict.keys():
            self._nodes[node.name] = node
            for toNode in graph_dict[node]:
                edge = Edge(node, toNode, EdgeDirection.ONEWAY)
                self._edges[edge.id] = edge

        self._build_maps()


    def _build_maps(self):
        self._pos_edge_map = self.__generate_pos_edge_map(self._edges)
        self._node_edge_map = self.__generate_node_edge_map(self._edges)
        self._pos_node_map = self.__generate_position_node_map(self._nodes)
        self._graph_dict = self.__generate_graph_dict(self._edges)
        self._node_by_name_map = self.__generate_node_by_name_map(self._nodes)

    def __generate_node_edge_map(self, edges: Dict[str, Edge]):
        node_edge_map = {}
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
            edges_by_pos_dict.setdefault(edge.start.pos, {})[edge.end.pos] = id

        return edges_by_pos_dict

    def __generate_position_node_map(self, nodes):
        pos_node_map = {}
        for id, node in nodes.items():
            pos = node.pos
            pos_node_map[pos] = id

        return pos_node_map


    def __generate_graph_dict(self, edges):
        graph_dict = {}
        for id, edge in edges.items():
            graph_dict.setdefault(edge.start, []).append(edge.end)

        for id, node in self._nodes.items():
            if node not in graph_dict.keys():
                graph_dict[node] = []
        return graph_dict

    def __generate_node_by_name_map(self, nodes):
        ret = {}
        for node_id, node in nodes.items():
            ret[node.name] = node_id
        return ret

    def nodes(self):
        """ returns the vertices of a graph """
        return copy.deepcopy([node for id, node in self._nodes.items()])

    def edges(self):
        """ returns the edges of a graph """
        return copy.deepcopy([edge for id, edge in self._edges.items()])

    def add_node(self, node):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if node.name not in self._nodes.keys():
            self._nodes[node.name] = node
        self._build_maps()

    def add_edges(self, edges):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        if isinstance(edges, list) and isinstance(edges[0], Edge):
            for edge in edges:
                self._add_edge(edge)
        elif isinstance(edges, dict) and isinstance(list(edges.keys())[0], IVector):
            for start in edges.keys():
                for end in edges[start]:
                    start_node = self.node_at(start)
                    end_node = self.node_at(end)
                    if start_node and end_node:
                        edge = Edge(start_node, end_node)
                        self._add_edge(edge)
        elif isinstance(edges, Edge):
            self._add_edge(edges)

    def remove_edges(self, edges):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!
        """
        print(f"len start {len(self._pos_edge_map)}")
        if isinstance(edges, list) and isinstance(edges[0], Edge):
            for edge in edges:
                self._remove_edge(edge)
        elif isinstance(edges, dict) and isinstance(list(edges.keys())[0], IVector):
            for start in edges.keys():
                for end in edges[start]:
                    start_node = self.node_at(start)
                    end_node = self.node_at(end)
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
        elif isinstance(edges, dict) and isinstance(list(edges.keys())[0], IVector):
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
        elif isinstance(edges, dict) and isinstance(list(edges.keys())[0], IVector):
            for start in edges.keys():
                for end in edges[start]:
                    edge = self._edge_at(start, end)
                    if edge:
                        edge.add_disabler(disabler)
        elif isinstance(edges, Edge):
            edges.add_disabler(disabler)

    def edges_to_node(self, node: Node):
        if not isinstance(node, Node):
            raise Exception(f"input must be of type {Node}, but {type(node)} was provided")
        node_edges = self._node_edge_map[node]
        return [self._edges[edge_id] for edge_id in node_edges]


    def disable_edges_to_node(self, node, disabler):
        logging.debug(f"disable {node} with disabler {disabler}")
        if isinstance(node, IVector):
            node = self.node_at(node)

        if isinstance(node, Node):
            node_edges = self._node_edge_map[node]
            for edge_id in node_edges:
                edge = self._edges[edge_id]
                edge.add_disabler(disabler)

    def enable_edges_to_node(self, node, disabler):
        logging.debug(f"enable {node} on disabler {disabler}")
        if isinstance(node, IVector):
            node = self.node_at(node)

        if isinstance(node, Node):
            node_edges = self._node_edge_map[node]
            for edge_id in node_edges:
                edge = self._edges[edge_id]
                edge.remove_disabler(disabler)

    def adjacent_nodes(self, node: Node, only_enabled: bool = False, ignored_disablers: set = None) -> List[Node]:
        adjacents = list(self._graph_dict[node])
        if only_enabled:
            adjacents[:] = [x for x in adjacents if self._edge_at(node.pos, x.pos).enabled(ignored_disablers)]

        return adjacents

    def _remove_edge(self, edge: Edge):
        edge = self._edge_at(edge.start.pos, edge.end.pos)
        if edge.id in self._pos_edge_map.keys():
            del self._edges[edge.id]
        self._build_maps()

    def _add_edge(self, edge: Edge):
        existing_edge = self._edge_at(edge.start.pos, edge.end.pos)
        if existing_edge is None:
            self._edges[edge.id] = edge
        self._build_maps()

    def node_at(self, pos: IVector):
        return self._nodes.get(self._pos_node_map.get(pos, None), None)

    def nodes_at(self, points: List[IVector]):
        nodes = [self.node_at(point) for point in points]
        return [node for node in nodes if node]

    def _edge_at(self, start: IVector, end: IVector):
        # start = self.node_at(start).pos
        # end = self.node_at(end).pos

        if not (start and end):
            return None

        edge_id = self._pos_edge_map.get(start, {}).get(end, None)
        if edge_id:
            return self._edges.get(edge_id, None)

        return None





    def __str__(self):
        res = "vertices: "
        for id, node in self._nodes.items():
            res += f"\n\t{str(node)}"
        res += "\nedges: "
        for edge in self.edges():
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
        nodes = self._nodes
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
        nodes = self._nodes
        path = path + [start_vertex]
        if start_vertex == end_vertex:
            return [path]
        if start_vertex.name not in nodes.keys() or end_vertex.name not in nodes.keys():
            return []
        paths = []
        for edge_id in self._node_edge_map[start_vertex]:
            edge = self._edges[edge_id]
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
        edges = [self._edges[x] for x in edges]
        degree = len(edges) + edges.count([x for x in edges if x.end == node])
        return degree


    def degree_sequence(self):
        """ calculates the degree sequence """
        seq = []
        for id, node in self._nodes.items():
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
        for id, node in self._nodes.items():
            vertex_degree = self.vertex_degree(node)
            if vertex_degree < min:
                min = vertex_degree
        return min


    def Delta(self):
        """ the maximum degree of the vertices """
        max = 0
        for id, node in self._nodes.items():
            vertex_degree = self.vertex_degree(node)
            if vertex_degree > max:
                max = vertex_degree
        return max


    def density(self):
        """ method to calculate the density of a graph """
        g = self._nodes
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

    def astar(self, start: Node, end: Node, g_func: Callable[[Node, Node], float] = None, h_func: Callable[[Node, Node], float] = None, ignored_disablers:List[str]=None) -> List[Node]:

        """Returns a list of tuples as a path from the given start to the given end in the given graph"""
        # print(f"Performing A* over map of length: {len(self.__graph_dict)}")

        # Create start and end node
        start_node = AStarNode(None, start)
        end_node = AStarNode(None, end)

        # Initialize both open and closed list
        open_set = set()
        closed_set = set()

        # Add the start node
        open_set.add(start_node)

        # Loop until you find the end
        while len(open_set) > 0:

            # Find the node on open list with the least F value
            current_node = next(iter(open_set))
            for open_item in open_set:
                if open_item.f < current_node.f:
                    current_node = open_item

            # Pop current off open list, add to closed list
            open_set.remove(current_node)
            closed_set.add(current_node)

            # Found the goal
            if current_node == end_node:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.graph_node)
                    current = current.parent
                return path[::-1]  # Return reversed path

            # Generate children
            children = []
            for new_graph_node in self._graph_dict[current_node.graph_node]:  # Adjacent nodes
                edge = self._edge_at(current_node.graph_node.pos, new_graph_node.pos)
                if edge.enabled() or all(elem in ignored_disablers for elem in edge.disablers()):
                    # Create new node
                    new_node = AStarNode(current_node, new_graph_node)

                    # Append
                    children.append(new_node)

            # Loop through children
            for child in children:
                # Child is on the closed list
                if child in closed_set:
                    continue

                # Create the f, g, and h values (use input functions to calculate g and h if provide, otherwise use distance between node positions)
                child.g = current_node.g + g_func(current_node.graph_node, child.graph_node) if g_func else child.graph_node.pos.distance_from(current_node.graph_node.pos)
                child.h = h_func(child.graph_node, end_node.graph_node) if h_func else child.graph_node.pos.distance_from(end_node.graph_node.pos)
                child.f = child.g + child.h

                # Add the child to the open list
                if child in open_set:
                    existing = open_set.remove(child)
                    if existing is not None and child.g > existing.g:
                        open_set.add(existing)
                    else:
                        open_set.add(child)
                else:
                    open_set.add(child)



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
        return self._nodes[self._node_by_name_map[node_name]]



    def verify_edge_configuration(self, edges_to_compare: List[Edge]):
        for edge in edges_to_compare:
            if not self._edges[self._pos_edge_map[edge.start.pos][edge.end.pos]].config_match(edge):
                return False

        return True

    '''A recursive function that find articulation points  
        using DFS traversal 
        u --> The vertex to be visited next 
        visited[] --> keeps tract of visited vertices 
        disc[] --> Stores discovery times of visited vertices 
        parent[] --> Stores parent vertices in DFS tree 
        ap[] --> Store articulation points'''

    def APUtil(self, u, visited, ap, parent, low, disc, iter):

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
                    ap[u] = True

                # (2) If u is not root and low value of one of its child is more
                # than discovery value of u.
                if parent[u] != -1 and low[v] >= disc[u]:
                    ap[u] = True

                    # Update low value of u for parent function calls
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

                # The function to do DFS traversal. It uses recursive APUtil()

    def AP(self):

        # Mark all the vertices as not visited
        # and Initialize parent and visited,
        # and ap(articulation point) arrays
        visited = {node: False for id, node in self._nodes.items()}
        disc = {node: float("Inf") for node in self._graph_dict.keys()}
        low = {node: float("Inf") for node in self._graph_dict.keys()}
        parent = {node: -1 for node in self._graph_dict.keys()}
        ap = {node: False for node in self._graph_dict.keys()}  # To store articulation points

        print(visited)
        print(disc)
        print(low)
        print(parent)
        print(ap)

        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for i in self._graph_dict.keys():
            if visited[i] == False:
                self.APUtil(i, visited, ap, parent, low, disc, 0)

        print(f"TESTTEST ap{ap}")
        for index, value in enumerate(ap):
            if value == True:
                print(index)