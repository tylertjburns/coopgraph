import matplotlib.pyplot as plt
import cooptools.plotting as cplt
from coopgraph.graphs import Graph, Node
from cooptools.colors import Color

def plot_graph(
        graph: Graph,
        ax,
        fig,
        color: Color = None
):
    #plot nodes
    pts = [x.pos for x in graph.nodes]
    cplt.plot_series(
        pts,
        ax=ax,
        fig=fig,
        series_type='scatter',
        color=color,
        point_size=1,
        labels=[x.name for x in graph.nodes]
    )

    #plot edges
    for edge in graph.edges:
        cplt.plot_series(
            points=[edge.start.pos, edge.end.pos],
            ax=ax,
            series_type='line',
            color=color,
            fig=fig,
        )