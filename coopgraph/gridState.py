class GridState():
    def __init__(self, state=None, start_toggled:bool = True):
        self.state = state if state else {}
        self.toggled = start_toggled

    def toggle(self):
        self.toggled = not self.toggled