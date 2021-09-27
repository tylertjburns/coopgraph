from typing import Dict

class GridState:
    def __init__(self, grid, row: int, column: int, state: Dict=None):
        self.state = state if state else {}
        self.row = row
        self.column = column
        self.grid = grid

    def copy(self):
        return GridState(self.grid, self.row, self.column, state=self.state)

    @property
    def left(self):
        if self.column > 0:
            return self.grid[self.row][self.column - 1]
        else:
            return None

    @property
    def right(self):
        if self.column < self.grid.nColumns - 1:
            return self.grid[self.row][self.column + 1]
        else:
            return None

    @property
    def up(self):
        if self.row > 0:
            return self.grid[self.row - 1][self.column]
        else:
            return None

    @property
    def down(self):
        if self.row < self.grid.nRows - 1:
            return self.grid[self.row + 1][self.column]
        else:
            return None

    def __repr__(self):
        return str(self.state)

    def __getitem__(self, item):
        return self.state[item]