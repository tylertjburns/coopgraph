import uvicorn

from coopgraph.graphs import Graph
from coopgraph.api_utils import edges_api_factory, nodes_api_factory
from fastapi import FastAPI

graph = Graph()

app = FastAPI()
node_router = nodes_api_factory(f'/nodes', graph)
edge_router = edges_api_factory('/edges', graph)

app.include_router(node_router.router)
app.include_router(edge_router.router)


uvicorn.run(app=app, port=1219)