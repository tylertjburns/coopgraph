from abc import ABC, abstractmethod
from coopgraph.gridState import GridState
from coopgraph.toggles import Toggleable
from typing import Type

class IOnGridSelectPolicy(ABC):
    def __init__(self, key: str):
        self.key = key

    @abstractmethod
    def act_on_gridstate(self, grid_state: GridState, **kwargs):
        pass

    @abstractmethod
    def initialize_state(self, grid_state: GridState):
        pass

    def __eq__(self, other):
        if issubclass(type(other), IOnGridSelectPolicy) and other.key == self.key:
            return True
        else:
            return False

    def __repr__(self):
        return f"{type(self)} with key [{self.key}]"

    def __hash__(self):
        return hash(self.key)

class DoNothingPolicy(IOnGridSelectPolicy):
    def __init__(self, key:str):
        IOnGridSelectPolicy.__init__(self, key)

    def act_on_gridstate(self, grid_state: GridState, **kwargs):
        pass

    def initialize_state(self, grid_state: GridState):
        pass

class TogglePolicy(IOnGridSelectPolicy):
    def __init__(self, key: str, toggle: Type[Toggleable]):
        IOnGridSelectPolicy.__init__(self, key)
        self.toggle = toggle

    def act_on_gridstate(self, grid_state: GridState, **kwargs):
        grid_state.state[self.key].toggle()

    def initialize_state(self, grid_state: GridState):
        grid_state.state[self.key] = grid_state.state.setdefault(self.key, self.toggle.copy())

class IncrementPolicy(IOnGridSelectPolicy):
    def __init__(self, key: str, starting_index: int = 0, step_size: int = 1):
        IOnGridSelectPolicy.__init__(self, key)
        self.starting_index = starting_index
        self.step_size = step_size

    def act_on_gridstate(self, grid_state: GridState, **kwargs):
        grid_state.state[self.key] = grid_state.state[self.key] + self.step_size

    def initialize_state(self, grid_state: GridState):
        grid_state.state[self.key] = self.starting_index

class SetValuePolicy(IOnGridSelectPolicy):
    def __init__(self, key: str, starting_index: int = 0, step_size: int = 1):
        IOnGridSelectPolicy.__init__(self, key)
        self.starting_index = starting_index
        self.step_size = step_size

    def act_on_gridstate(self, grid_state: GridState, **kwargs):
        if 'value' not in kwargs.keys():
            raise Exception(f"Value must be provided for act_on_gridstate on {type(self)}")
        grid_state.state[self.key] = kwargs['value']

    def initialize_state(self, grid_state: GridState):
        grid_state.state[self.key] = None