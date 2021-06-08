from coopgraph.grids import GridSystem
from coopgraph.gridSelectPolicies import IncrementPolicy, TogglePolicy
from coopgraph.toggles import BooleanToggleable




increment_counter_policy = IncrementPolicy(key="counter")
toggle_bool_policy = TogglePolicy(key='toggle', toggle=BooleanToggleable(default=False))
toggle_left = TogglePolicy(key='left', toggle=BooleanToggleable(default=False))
toggle_right = TogglePolicy(key='right', toggle=BooleanToggleable(default=False))
toggle_up = TogglePolicy(key='up', toggle=BooleanToggleable(default=False))
toggle_down = TogglePolicy(key='down', toggle=BooleanToggleable(default=False))

grid = GridSystem(4, 4, grid_select_policies=[
                                                increment_counter_policy,
                                                toggle_bool_policy,
                                                toggle_left,
                                                toggle_right,
                                                toggle_up,
                                                toggle_down
                                            ])

def click(row, column):
    grid.act_on_grid(row, column, [
        increment_counter_policy,
        toggle_bool_policy,
        ])

    # Handle Left
    if column > 0:
        grid.act_on_grid(row, column - 1, [toggle_right])

    # Handle Up
    if row > 0:
        grid.act_on_grid(row - 1, column, [toggle_down])

    # Handle Right
    if column < grid.nColumns - 1:
        grid.act_on_grid(row, column + 1, [toggle_left])

    # Handle Down
    if row < grid.nRows - 1:
        grid.act_on_grid(row + 1, column, [toggle_up])


# print(grid.grid[1][1].state)
print(grid)
click(1, 1)
print(grid)
click(1, 1)
print(grid)
click(1, 1)
print(grid)

print(grid.state_value_as_array(key='toggle'))
print(grid.state_value_as_array(key='counter'))


# print(grid.grid[1][1].state)
# grid.act_on_grid(1, 1, [increment_counter_policy, toggle_bool_policy])
# print(grid.grid[1][1].state)
# grid.act_on_grid(1, 1, [increment_counter_policy])
# print(grid.grid[1][1].state)







# astar = grid.astar(grid.nodes_at_point(Vector2(0, 0))[0], grid.nodes_at_point(Vector2(5, 5))[0])

# pprint.pprint(astar.path)
# pprint.pprint(astar.steps)
