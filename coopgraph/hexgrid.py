import hexy as hx
from coopgraph.graphs import Graph
from coopgraph.IGridSystem import IGridSystem
import numpy as np
from coopstructs.vectors import Vector2, IVector
from coopstructs.geometry import Rectangle
from typing import Dict
from coopgraph.graphs import Node

class HexGridSystem(IGridSystem):
    def __init__(self, axial_coordinates: np.array(Vector2), hex_radius: float):
        self.hex_radius = hex_radius
        self.hex_map = hx.HexMap()

        hexes = []
        for i, axial in enumerate(axial_coordinates):
            hexes.append(hx.HexTile(axial, hex_radius, hash(axial)))


        self.hex_map[np.array(axial_coordinates)] = hexes

        IGridSystem.__init__(self, axial_coordinates.shape[0], axial_coordinates.shape[1], )

    def build_a_graph_dict(self, pos_node_map: Dict[IVector, Node]):
        pass

    def build_position_node_map(self, ncols: int, nrows: int):
        pass

    def coord_from_grid_pos(self):
        pass

    def grid_from_coord(self, x, y, area_rect: Rectangle):
        pass

    def grid_unit_height(self):
        return self.hex_radius * 2

    def grid_unit_width(self):
        return self.hex_radius * 2

if __name__ == "__main__":
    # spiral_coordinates = hx.get_spiral(np.array((0, 0, 0)), 1, 6)
    # print(spiral_coordinates)
    # axial_coordinates = hx.cube_to_axial(spiral_coordinates)
    # print(axial_coordinates)
    
    axial_coords = []
    for ii in range(0, 5):
        for jj in range(0, 5):
            axial_coords.append(Vector2(ii, jj))

    hexgrid = HexGridSystem(np.array(axial_coords), 5)


    