from coopstructs.vectors import Vector2, IVector
from coopgraph.grids import GridSystem
import pprint
from coopgraph.gridSelectPolicies import IncrementPolicy, TogglePolicy

grid = GridSystem(10, 10, grid_select_policies=[IncrementPolicy(), TogglePolicy()])

astar = grid.astar(grid.node_at(Vector2(0, 0)), grid.node_at(Vector2(5, 5)))

pprint.pprint(astar.path)
pprint.pprint(astar.steps)

grid.select_grid(Vector2(1, 1))

print(grid.grid[1][1].state, grid.grid[1][1].toggled)
grid.select_grid(Vector2(1, 1))
print(grid.grid[1][1].state, grid.grid[1][1].toggled)

x = Vector2(1, 1)
print(isinstance(x, IVector))