from abc import ABC
from typing import Dict, List
from coopgraph.graphs import Node

class IGraph(ABC):
    def __init__(self, graph_dict: Dict[Node, List[Node]] = None):
        pass