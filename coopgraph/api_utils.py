from coopapi.apiShell import ApiShell
from typing import List
from pydantic.dataclasses import dataclass
from fastapi import Request
from coopgraph.graphs import Graph, Node, Edge
from typing import Dict

def _edge_dict(edge: Edge):
    return {
        'id': edge.id,
        'start_node_id': edge.start.name,
        'end_node_id': edge.end.name,
        'direction': edge.direction.name,
        'disablers': edge.disablers(),
        'length': edge.length
    }

@dataclass
class API_Node:
    id: str
    pos: Dict
    edges: List[Dict]

    @staticmethod
    def from_Node(node: Node, graph: Graph):
        return API_Node(id=node.name, pos=node.pos, edges=[_edge_dict(x) for x in graph.edges_to_node(node)])

    def to_node(self):
        return Node(self.id, (self.pos['x'], self.pos['y']))

@dataclass
class API_Edge:
    id: str
    start_node_id: str
    end_node_id: str
    weight: float = None

    @staticmethod
    def from_Edge(edge: Edge):
        return API_Edge(id=edge.id, start_node_id=edge.start.name, end_node_id=edge.end.name, weight=edge.weight)

    @staticmethod
    def from_dict(dict: Dict):
        return API_Edge(**dict)

    def to_edge(self, graph: Graph):
        return graph.edges_by_id([self.id])[0]

def new_node(graph: Graph,
             request: Request,
             new: API_Node):
    graph.add_node(new.to_node())
    graph.add_edges([API_Edge.from_dict(x).to_edge(graph) for x in new.edges])

    return API_Node.from_Node(new.to_node(), graph)

def get_nodes(request: Request,
              graph: Graph):
    ret = [API_Node(x.name, x.pos.coords, [_edge_dict(e) for e in graph.edges_to_node(x)]) for x in graph.nodes]
    return ret

def nodes_api_factory(base_url_route: str,
                      graph: Graph) -> ApiShell:

    on_post = lambda req, t: new_node(graph=graph, request=req, new=t)
    on_getmany = lambda req, q, l: get_nodes(graph=graph, request=req)
    return ApiShell(
        target_schema=API_Node,
        base_route=base_url_route,
        on_post_callback=on_post,
        on_getmany_callback=on_getmany
    )



def new_edge(graph: Graph,
             request: Request,
             new: API_Edge):
    graph.add_edges([Edge(nodeA=graph.node_by_name(node_name=new.start_node_id),
                          nodeB=graph.node_by_name(node_name=new.end_node_id),
                          edge_weight=new.weight,
                          naming_provider=lambda: new.id
                          )])

    added = graph.edges_by_id([new.id])[0]
    return API_Edge.from_edge(added)

def get_edges(request: Request,
              graph: Graph):
    ret = [API_Edge(x.name, x.pos.coords, [_edge_dict(e) for e in graph.edges_to_node(x)]) for x in graph.nodes]
    return ret

def edges_api_factory(base_url_route: str,
                      graph: Graph) -> ApiShell:

    on_post = lambda req, t: new_node(graph=graph, request=req, new=t)
    on_getmany = lambda req, q, l: get_nodes(graph=graph, request=req)
    return ApiShell(
        target_schema=API_Node,
        base_route=base_url_route,
        on_post_callback=on_post,
        on_getmany_callback=on_getmany
    )


# @dataclass
# class API_Graph:
#     graph_dict: Dict
#
#
# def graph_api_factory(base_url_route: str,
#                       graph: Graph) -> ApiShell:
#
#     on_post = lambda req, t: graph.
#     on_getmany = lambda req, q, l: get_nodes(graph=graph, request=req)
#     return ApiShell(
#         target_schema=API_Graph,
#         base_route=base_url_route,
#         on_post_callback=on_post,
#         on_getmany_callback=on_getmany
#     )
