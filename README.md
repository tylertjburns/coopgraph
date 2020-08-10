# coopgraph
Logical Graph Builder that can be used for various problems that can be modeled as a graph data structure

An Example:
```
from Graphs import Graph, Node
from dataStructs import Vector2

a = Node(name='A', pos=Vector2(0, 0))
b = Node(name='B', pos=Vector2(3, 3))
c = Node(name='C', pos=Vector2(2, 0))
d = Node(name='D', pos=Vector2(2, 1))
e = Node(name='E', pos=Vector2(3, 4))
f = Node(name='F', pos=Vector2(5, 5))


g = { a: [d],
      b: [c],
      c: [b, d, e],
      d: [a, c],
      e: [c, f],
      f: []
    }

graph = Graph(g)
```

The graph structure can then be used to perform various graph-related analysis:

Two find nodes that have no outbound connections 
```
print(graph.find_isolated_vertices())
```

To find the shortest path between two nodes
```
print(graph.astar(a, e))
```

Note that for astar calculation, edges can be enabled or disabled against one or more disablers. This is useful for implementing temporary criteria in:
```
edges_to_disable = [value for key, value in graph.edges()][:3]

graph.disable_edges(edges_to_disable, "myDisabler")
path = graph.astar(a, e)
graph.disable_edges(edges_to_disable, "myDisabler")
```

you can also ignore disablers directly by passing a list of disabler names to the astar() method
```
edges_to_disable = [value for key, value in graph.edges()][:3]
graph.disable_edges(edges_to_disable, "myIngoredDisabler")

ignored = ["myIngoredDisabler"]
path = graph.astar(a, e, ignored_disablers=ignored)
```

An astar() call can also include custom g and h functions that allow for better control of the astar algorithm
```
def g(node1 Node, node2: Node) -> float:
    if node1.pos - node2.pos > 10:
        return 1
    else
        return .5

def h(node1 Node, node2: Node) -> float:
    if node1.pos - node2.pos > 10:
        return 100
    else
        return -100

path = graph.astar(a, e, g_func=g, h_func=h)
```
