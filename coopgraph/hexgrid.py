import hexy as hx
from coopgraph.graphs import Node
from coopgraph.AGrid import AGrid
import numpy as np
from coopstructs.vectors import Vector2, IVector
from coopstructs.geometry import Rectangle
from typing import Dict
from coopgraph.dataStructs import GridPoint


class HexGridSystem(AGrid):
    def __init__(self, axial_coordinates: np.array(Vector2), hex_radius: float):
        self.hex_radius = hex_radius
        self.hex_map = hx.HexMap()

        hexes = []
        coords = []
        for i, axial in enumerate(axial_coordinates):
            coords.append((axial.x, axial.y))
            hexes.append(hx.HexTile((axial.x, axial.y), hex_radius, hash(axial)))

        self.hex_map[np.array(coords)] = hexes

        columns = max(vec.x for vec in axial_coordinates) - min(vec.x for vec in axial_coordinates)
        rows = max(vec.y for vec in axial_coordinates) - min(vec.y for vec in axial_coordinates)

        AGrid.__init__(self, rows, columns)

    def coord_from_grid_pos(self, grid_pos: Vector2, area_rect: Rectangle, grid_point: GridPoint = GridPoint.CENTER):
        pass

    def grid_from_coord(self, grid_pos: Vector2, area_rect: Rectangle):
        pass

    def grid_unit_height(self, area_rect: Rectangle):
        return self.hex_radius * 2

    def grid_unit_width(self, area_rect: Rectangle):
        return self.hex_radius * 2

    def build_graph_dict(self, pos_node_map: Dict[IVector, Node], **kwargs):
        graph_dict = {}

        for pos in pos_node_map.keys():
            graph_dict[pos_node_map[pos]] = []
            connections = [
                Vector2(pos.x - 1, pos.y) if pos_node_map.get(Vector2(pos.x - 1, pos.y), None) else None,  # left
                Vector2(pos.x + 1, pos.y) if pos_node_map.get(Vector2(pos.x + 1, pos.y), None) else None,  # right
                Vector2(pos.x, pos.y - 1) if pos_node_map.get(Vector2(pos.x, pos.y - 1), None) else None,  # up
                Vector2(pos.x, pos.y + 1) if pos_node_map.get(Vector2(pos.x, pos.y + 1), None) else None,  # down
            ]

            for connection_pos in connections:
                try:
                    if connection_pos:
                        graph_dict[pos_node_map[pos]].append(pos_node_map[connection_pos])
                except:
                    print(f"{connection_pos} \n"
                          f"{pos_node_map}")
                    print(f"connection pos: {type(connection_pos)}")
                    print(f"first pos_node_map pos: {type([x for x in pos_node_map.keys()][0])}")
                    raise
        return graph_dict


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

