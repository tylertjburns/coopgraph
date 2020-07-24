from typing import List
from enum import Enum
import math


class GridPoint(Enum):
    ORIGIN = 1
    CENTER = 2

class IVector:
    def __init__(self):
        self.coords = {}
    @property
    def x(self):
        return self.coords.get('x', None)

    @property
    def y(self):
        return self.coords.get('y', None)

    @property
    def z(self):
        return self.coords.get('z', None)

    def unit(self):
        length = self.length()
        if length == 0:
            return None
        return self / self.length()

    def length(self):
        sum = 0
        for ii in self.coords:
            sum += self.coords[ii] ** 2

        return math.sqrt(sum)

    def distance_from(self, other):
        if isinstance(other, IVector):
            sum = 0.0
            for ii in self.coords.keys():
                delta = (float(self.coords[ii]) - float(other.coords[ii])) ** 2
                sum += delta
            return math.sqrt(sum)
        else:
            raise TypeError(f"type {other} cannot be distanced from {type(self)}")

    def __eq__(self, other) -> bool:
        if not (isinstance(other, type(self)) or issubclass(type(other), type(self)) or issubclass(type(self), type(other))):
            return False
        for ii in self.coords.keys():
            if not self._is_close(self.coords[ii], other.coords[ii]):
                return False

        return True

    def __str__(self, n_digits: int = 2):
        ret = "<"
        ii = 0
        for key in self.coords.keys():
            if ii > 0:
                ret += ", "
            ret += f"{round(float(self.coords[key]), n_digits)}"
            ii +=1

        ret +=">"
        return ret

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))

    def __add__(self, other):
        if not (isinstance(other, type(self)) or issubclass(type(other), type(self)) or issubclass(type(self),
                                                                                                   type(other))):
            raise TypeError(f"Object of type [{type(other)}] cannot be added to {type(self)}")

        ret = IVector()
        for ii in self.coords:
            ret.coords[ii] = self.coords[ii] + other.coords[ii]

        return ret

    def __sub__(self, other):
        if not (isinstance(other, type(self)) or issubclass(type(other), type(self)) or issubclass(type(self), type(other))):
            raise TypeError(f"Object of type [{type(other)}] cannot be subtracted from {type(self)}")

        ret = IVector()
        for ii in self.coords:
            ret.coords[ii] = self.coords[ii] - other.coords[ii]

        return ret

    def __mul__(self, other):
        if not (isinstance(other, float) or isinstance(other, int)):
            raise TypeError(f"Object of type [{type(other)}] cannot be multiplied to {type(self)}")

        new_vector = IVector()
        for ii in self.coords:
            new_vector.coords[ii] = self.coords[ii] * other

        return new_vector

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not (isinstance(other, float) or isinstance(other, int)):
            raise TypeError(f"Object of type [{type(other)}] cannot be divided from {type(self)}")

        new_vector = IVector()
        for ii in self.coords:
            new_vector.coords[ii] = self.coords[ii] / other

        return new_vector
        # return Vector2(self.x / other, self.y / other)

    def _is_close(self, a:float, b:float, rel_tol=1e-09, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def dot(self, other):
        if not (isinstance(other, type(self)) or issubclass(type(other), type(self)) or issubclass(type(self), type(other))):
            raise TypeError(f"type {other} cannot be dot multiplied by {type(self)}. Must match type...")
        sum = 0
        for ii in self.coords:
            sum += self.coords[ii] * other.coords[ii]
        return sum

    def project_onto(self, other):
        if not (isinstance(other, type(self)) or issubclass(type(other), type(self)) or issubclass(type(self), type(other))):
            raise TypeError(f"type {other} cannot be projected onto {type(self)}. Must match type...")

        dot = self.dot(other)
        return other.unit() * dot / other.length()

    def hadamard_product(self, other):
        if not (isinstance(other, type(self)) or issubclass(type(other), type(self)) or issubclass(type(self), type(other))):
            raise TypeError(f"type {other} cannot be hadamard multiplied by {type(self)}. Must match type...")
        else:
            new_vector = IVector()
            for ii in self.coords:
                new_vector.coords[ii] = self.coords[ii] * other.coords[ii]
            return new_vector

    def hadamard_division(self, other, num_digits=None):
        if not (isinstance(other, type(self)) or issubclass(type(other), type(self)) or issubclass(type(self), type(other))):
            raise TypeError(f"type {type(other)} cannot be hadamard divided by {type(self)}. Must match type...\n"
                            f"{issubclass(type(other), type(self))}\n"
                            f"{issubclass(type(self), type(other))}")

        else:
            new_vector = IVector()
            for ii in self.coords:
                if num_digits is not None:
                    new_vector.coords[ii] = round(self.coords[ii] * 1.0 / other.coords[ii], num_digits)
                else:
                    new_vector.coords[ii] = self.coords[ii] * 1.0 / other.coords[ii]
            return new_vector

    def bounded_by(self, a, b) -> bool:
        if not isinstance(a, type(self)) and isinstance(b, type(self)):
            raise TypeError(f"a and b must match vector type: {type(self)}. {type(a)} and {type(b)} were given")

        for ii in self.coords.keys():
            min_val = min(a.coords[ii], b.coords[ii])
            max_val = max(a.coords[ii], b.coords[ii])

            if not min_val <= self.coords[ii] <= max_val:
                return False

        return True

class Vector2 (IVector):
    def __init__(self, x: float, y: float):
        IVector.__init__(self)
        self.coords['x'] = x
        self.coords['y'] = y

    def as_tuple(self):
        return (self.x, self.y)


class Rectangle:
    def __init__(self, x, y, height, width):
        self.x = x
        self.y = y
        self.height = height
        self.width = width


    def __str__(self):
        return f"TopLeft: <{self.x}, {self.y}>, Size: H{self.height} x W{self.width}"


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



if __name__ == "__main__":
    a = Vector2(1, 2)
    b = Vector2(1, 3)
    c = Vector2(1, 4)
    d = Vector2(1, 2)
    e = Vector2(1.0, 2.0)

    f = IVector()
    f.coords['x'] = 1
    f.coords['y'] = 2

    g = 1.0
    h = 1.0

    i = {
        a: "a",
        b: "b",
        c: "c"
    }

    assert (a * 4) == Vector2(4, 8)

    assert b.bounded_by(a, c) is True, f"{b} should be bounded by {a, c}"
    assert a.bounded_by(b, c) is False, f"{a} should not be bounded by {b, c}"
    assert c.bounded_by(a, c) is True, f"{c} should be bounded by {a, c}"

    assert (a == b) is False, f"{a} should not equal {b}"
    assert (a == d) is True, f"{a} should equal {d}"
    assert (a == e) is True, f"{a} should equal {e}"

    assert (a.distance_from(b)) == 1, f"distance from {a} to {b} should be 1"

    assert (c.hadamard_division(a)) == Vector2(1, 2), f"hadamard division of {c} and {a} should be {Vector2(1, 2)}"
    assert (c.hadamard_division(a)) == Vector2(1, 2), f"hadamard division of {c} and {a} should be {Vector2(1, 2)}"

    assert f._is_close(g, h) is True, f"{g} and {h} should be close"
    assert issubclass(type(a), IVector) is True, f"{type(a)} should be subclass of {type(IVector)}"
    assert (f == a) is True, f"IVector and Vector2 should be equal for {f} and {a}"

    assert (i[f] == 'a')