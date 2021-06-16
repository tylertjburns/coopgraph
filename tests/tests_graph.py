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

        assert len(graph.nodes), 6
        assert len(graph.edges), 9

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

        graph = Graph(g)

        assert len(graph.nodes) == 6
        assert graph.nodes == [a, b, c, d, e, f]


    def test_edges(self):
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

        graph = Graph(g)

        assert len(graph.edges) == 9
        assert graph.edges == [graph.edge_between(a, d), graph.edge_between(b, c), graph.edge_between(c, b), graph.edge_between(c, d), graph.edge_between(c, e), graph.edge_between(d, a), graph.edge_between(d, c), graph.edge_between(e, c), graph.edge_between(e, f)]

    def test_add_node(self):
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

        graph = Graph(g)

        assert len(graph.nodes) == 6
        assert graph.nodes == [a, b, c, d, e, f]

        h = Node(name='H', pos=Vector2(6, 5))
        graph.add_node(h)

        assert len(graph.nodes) == 7
        assert graph.nodes == [a, b, c, d, e, f, h]

    def test_add_edges(self):
        pass

    def test_remove_edges(self):
        pass

    def test_enable_edges(self):
        pass

    def test_disable_edges(self):
        a = Node(name='A', pos=Vector2(0, 0))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, b, c],
             e: [c, f],
             f: []
             }

        graph = Graph(g)
        graph.disable_edges(graph.edge_between(d, c), "BLOCK")
        assert graph.edge_between(d, c).disablers() == {"BLOCK"}


    def test_edges_to_node(self):
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

        graph = Graph(g)

        assert set(graph.edges_to_node(c)) == {graph.edge_between(b, c), graph.edge_between(d, c), graph.edge_between(e, c), graph.edge_between(c, b), graph.edge_between(c, d), graph.edge_between(c, e)}

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

    def test_nodes_at__none(self):
        a = Node(name='A', pos=Vector2(1, 1))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, c],
             e: [c],
             f: []
             }

        graph = Graph(g)

        nodes = graph.nodes_at_point(Vector2(0, 0))

        assert nodes == []

    def test_nodes_at__single(self):
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
             e: [c],
             f: []
             }

        graph = Graph(g)

        nodes = graph.nodes_at_point(Vector2(0, 0))

        assert nodes == [a]

    def test_nodes_at__multiple(self):
        a = Node(name='A', pos=Vector2(0, 0))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))
        h = Node(name='H', pos=Vector2(0, 0))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, c],
             e: [c],
             f: [],
             h: []
             }

        graph = Graph(g)

        nodes = graph.nodes_at_point(Vector2(0, 0))

        assert nodes == [a, h]


    def test__edge_at(self):
        pass

    def test_find_isolated_vertices(self):
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
             e: [c],
             f: []
             }

        graph = Graph(g)
        assert graph.find_isolated_vertices() == [f]

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

        graph = Graph(g)
        path = graph.astar(a, e)


        assert path.path == [a, d, c, e]

    def test__path_length(self):
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

        graph = Graph(g)
        path = [a, d, c, e]
        length = graph.path_length(path)

        assert length == d.pos.distance_from(a.pos) + c.pos.distance_from(d.pos) + e.pos.distance_from(c.pos)

    def test__astar_with_disablers(self):
        a = Node(name='A', pos=Vector2(0, 0))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, b, c],
             e: [c, f],
             f: []
             }

        graph = Graph(g)
        graph.disable_edges(graph.edge_between(d, c), "BLOCK")

        path = graph.astar(a, e)

        assert path.path == [a, d, b, c, e]

    def test_astar_with_ignored_disablers(self):
        a = Node(name='A', pos=Vector2(0, 0))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, b, c],
             e: [c, f],
             f: []
             }

        graph = Graph(g)
        edge = graph.edge_between(d, c)
        graph.disable_edges(edge, "BLOCK")

        path = graph.astar(a, e, ignored_disablers=["BLOCK"])

        assert path.path == [a, d, c, e]

    def test_astar_no_valid_path(self):
        a = Node(name='A', pos=Vector2(0, 0))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, b, c],
             e: [c],
             f: []
             }

        graph = Graph(g)
        path = graph.astar(a, f)

        assert path.path is None


    def test_node_by_name(self):
        pass

    def test_verify_edge_configuration(self):
        pass

    def test_APUtil(self):
        pass

    def test_AP(self):
        pass


    def test_add_node_with_connnections__base(self):
        a = Node(name='A', pos=Vector2(0, 0))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, b, c],
             e: [c],
             f: []
             }

        graph = Graph(g)


        h = Node('H', Vector2(100, 100))

        graph.add_node_with_connnections(h, {a: EdgeDirection.FROM, b: EdgeDirection.TO, c: EdgeDirection.TWOWAY})


        assert h in graph.nodes

        assert graph.edge_between(a, h) is not None
        assert graph.edge_between(h, a) is None
        assert graph.edge_between(b, h) is None
        assert graph.edge_between(h, b) is not None
        assert graph.edge_between(c, h) is not None
        assert graph.edge_between(h, c) is not None

    def test_closest_node__base(self):
        a = Node(name='A', pos=Vector2(0, 0))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, b, c],
             e: [c],
             f: []
             }

        graph = Graph(g)

        test = Vector2(0, 1)
        closest = graph.closest_nodes(test)[0]
        assert closest == a; f"{closest}"

    def test_copy__base(self):
        a = Node(name='A', pos=Vector2(0, 0))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, b, c],
             e: [c],
             f: []
             }

        graph = Graph(g)

        copy = graph.copy()

        assert len(copy.nodes) == len(graph.nodes)
        assert len(copy.edges) == len(graph.edges)
        assert (edge.disablers() == graph.edge_between(edge.start, edge.end).disablers() for edge in copy.edges)

    def test__edge_by_id(self):
        a = Node(name='A', pos=Vector2(0, 0))
        b = Node(name='B', pos=Vector2(3, 3))
        c = Node(name='C', pos=Vector2(2, 0))
        d = Node(name='D', pos=Vector2(2, 1))
        e = Node(name='E', pos=Vector2(3, 4))
        f = Node(name='F', pos=Vector2(5, 5))

        g = {a: [d],
             b: [c],
             c: [b, d, e],
             d: [a, b, c],
             e: [c],
             f: []
             }

        graph = Graph(g)

        edge = graph.edge_between(a, d)

        self.assertIsNotNone(edge, "Edge was returned none when expected value")

        edge_ret = graph.edges_by_id([edge.id])[0]

        self.assertEqual(edge, edge_ret, f"returned edge {edge_ret} [{edge_ret.id}] does not match {edge} [{edge.id}]")


if __name__ == "__main__":
    tester = TestGraph()
    tester.test_closest_node__base()