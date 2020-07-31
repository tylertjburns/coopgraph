from typing import List
from enum import Enum
from coopstructs.vectors import Vector2


class GridPoint(Enum):
    ORIGIN = 1
    CENTER = 2

class UnitPathDefinition:
    def __init__(self, toggled: int = 0, left: int = 0, up: int = 0, right: int = 0, down: int = 0):
        self.toggled = toggled
        self.left = left
        self.up = up
        self.right = right
        self.down = down

    def __str__(self):
        return f"L{self.left}-U{self.up}-R{self.right}-D{self.down}"

    def connection_count(self):
        return self.left + self.up + self.right + self.down

    def coord_at_extremities(self, origin: Vector2, path_box_size: Vector2) -> List[Vector2]:
        left = None
        top = None
        right = None
        bottom = None

        if self.left:
            left = Vector2(origin.x, int(origin.y + path_box_size.y / 2))

        if self.right:
            right = Vector2(origin.x + path_box_size.x, int(origin.y + path_box_size.y / 2))

        if self.up:
            top = Vector2(int(origin.x + path_box_size.x / 2), origin.y)

        if self.down:
            bottom = Vector2(int(origin.x + path_box_size.x / 2), int(origin.y + path_box_size.y))

        return [left, top, right, bottom]


