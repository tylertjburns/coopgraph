from abc import ABC, abstractmethod
from coopgraph.gridState import GridState

class IOnGridSelectPolicy(ABC):
    @abstractmethod
    def on_select(self, grid_state: GridState):
        pass

class DoNothingPolicy(IOnGridSelectPolicy):
    def on_select(self, grid_state: GridState):
        pass


class TogglePolicy(IOnGridSelectPolicy):
    def on_select(self, grid_state: GridState):
        grid_state.toggle()


class IncrementPolicy(IOnGridSelectPolicy):
    def on_select(self, grid_state: GridState):
        key = "counter"
        grid_state.state[key] = grid_state.state.get(key, 0) + 1