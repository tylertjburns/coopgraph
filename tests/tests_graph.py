import unittest
from coopgraph.graphs import Graph, Node, Edge, EdgeDirection
from coopgraph.dataStructs import Vector2

class TestGraph(unittest.TestCase):

    def init_a_test_graph(self):
        a = Node(name='A', pos=Vector2(0, 0))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, c],
             e: [c, f],
             f: []
             }

        return Graph(g)

    def test_init_graph(self):
        graph = self.init_a_test_graph()

        assert len(graph.nodes()), 6
        assert len(graph.edges()), 9

    def test__generate_node_edge_map(self):
        pass

    def test__generate_pos_edge_map(self):
        pass

    def test__generate_position_node_map(self):
        pass

    def test__generate_graph_dict(self):
        pass

    def test__generate_node_by_name_map(self):
        pass

    def test_nodes(self):
        graph = self.init_a_test_graph()

        assert graph.nodes() == {}
        pass

    def test_edges(self):
        pass

    def test_add_node(self):
        pass

    def test_add_edges(self):
        pass

    def test_remove_edges(self):
        pass

    def test_enable_edges(self):
        pass

    def test_disable_edges(self):
        pass

    def test_edges_to_node(self):
        pass

    def test_disable_edges_to_node(self):
        pass

    def test_enable_edges_to_node(self):
        pass

    def test_adjacent_nodes(self):
        pass

    def test__remove_edge(self):
        pass

    def test__add_edge(self):
        pass

    def test_node_at(self):
        pass

    def test_nodes_at(self):
        pass

    def test__edge_at(self):
        pass

    def test_find_isolated_vertices(self):
        pass

    def test_find_path(self):
        pass

    def test_find_all_paths(self):
        pass

    def test_vertex_degree(self):
        pass

    def test_degree_sequence(self):
        pass

    def test_is_degree_sequence(self):
        pass

    def test_delta(self):
        pass

    def test_Delta(self):
        pass

    def test_density(self):
        pass

    def test_diameter(self):
        pass

    def test_erdoes_gallai(self):
        pass

    def test_astar(self):
        pass

    def test_path_length(self):
        pass

    def test_node_by_name(self):
        pass

    def test_verify_edge_configuration(self):
        pass

    def test_APUtil(self):
        pass

    def test_AP(self):
        pass



    # def test_vertex_degree(self):

#
# a = Node(name='A', pos=Vector2(0, 0))
# b = Node(name='B', pos=Vector2(3, 3))
# c = Node(name='C', pos=Vector2(2, 0))
# d = Node(name='D', pos=Vector2(2, 1))
# e = Node(name='E', pos=Vector2(3, 4))
# f = Node(name='F', pos=Vector2(5, 5))
#
#
# g = { a: [d],
#       b: [c],
#       c: [b, d, e],
#       d: [a, c],
#       e: [c, f],
#       f: []
#     }
#
# graph = Graph(g)
# print(graph)
#
# graph.AP()
#
# print("Vertex Degrees")
# for node in graph.nodes():
#     print(f"degree of {node}: {graph.vertex_degree(node)}")
#
# print("List of isolated vertices:")
# print(graph.find_isolated_vertices())
#
# print("""A path from "a" to "e":""")
# print(graph.find_path(a, e))
#
# print("""All paths from "a" to "e":""")
# print(graph.find_all_paths(a, e))
#
# print("The maximum degree of the graph is:")
# print(graph.Delta())
#
# print("The minimum degree of the graph is:")
# print(graph.delta())
#
# print("Edges:")
# print(graph.edges())
#
# print("Degree Sequence: ")
# ds = graph.degree_sequence()
# print(ds)
#
# fullfilling = [ [2, 2, 2, 2, 1, 1],
#                      [3, 3, 3, 3, 3, 3],
#                      [3, 3, 2, 1, 1]
#                    ]
# non_fullfilling = [ [4, 3, 2, 2, 2, 1, 1],
#                     [6, 6, 5, 4, 4, 2, 1],
#                     [3, 3, 3, 1] ]
#
# for sequence in fullfilling + non_fullfilling :
#     print(sequence, Graph.erdoes_gallai(sequence))
#
# print("Add vertex 'x':")
# x = Node('X', Vector2(0, 0))
# graph.add_node(x)
# print(graph)
#
# print("Add vertex 'y':")
# y = Node('Y', Vector2(0, 0))
# graph.add_node(y)
# print(graph)
#
#
# print("Add edge ('x','y'): ")
# graph.add_edges([Edge(x, y)])
# print(graph)
#
# print("Add edge ('a','d'): ")
# graph.add_edges([Edge(a, d)])
# print(graph)
#
# print ("A* from 'a' to 'e'")
# path = graph.astar(a, e)
# length = graph.path_length(path)
# print(f"{path} with length {length}")