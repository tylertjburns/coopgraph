# Add icons folder to working directory
import sys, os, getopt

# Import and initialize the pygame library
import pygame
from coopgraph.graphs import Graph, Edge, Node
from coopstructs.vectors import Vector2, IVector
import csv
from typing import Dict

class SprNode(pygame.sprite.Sprite):
    def __init__(self, center: (int, int), size: (int, int)):
        if issubclass(type(center), IVector):
            center = (center.x, center.y)

        super(SprNode, self).__init__()
        # Create an image of the block, and fill it with a color.
        self.image = pygame.Surface([size[0], size[1]])
        self.image.fill((255, 0, 0))

        # Fetch the rectangle object that has the dimensions of the image
        # Update the position of this object by setting the values of rect.x and rect.y

        self.surf = pygame.Surface((size[0], size[1]))
        self.surf.fill((255, 255, 255))
        self.rect = self.surf.get_rect(
            center = center
        )



# Import pygame.locals for easier access to key coordinates
# Updated to conform to flake8 and black standards
from pygame.locals import (
    RLEACCEL,
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

# Initialize pygame
pygame.init()

# Define constants for the screen width and height
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1000

DATA_SCALE_X = 100
DATA_SCALE_Y = 50

# Define Environmental Constants
NODE_SIZE = 10

# Define Colors
RED = (230, 30, 30)


def read_2d_points_from_data_file(fileName: str):
    points = {}
    with open(fileName) as csvfile:
        data = csv.DictReader(csvfile)

        for row in data:
            points[row["name"]] = Vector2(float(row["x"]), float(row["y"]))

    return points

def create_a_walk(points: Dict[str, IVector]) -> ({str, Node}, {str, Edge}):
        g = { }

        old_point = None
        new_point = None
        for name, point in points.items():
            new_point = Node(name, point)

            if old_point is not None:
                g.setdefault(old_point, []).append(new_point)
            old_point = new_point

        g[new_point] = []
        return Graph(g)

def create_node_sprites(graph: Graph):
    nodes = graph.nodes()
    new_nodes = []
    for node in nodes:
        position = node.pos
        scale = Vector2(DATA_SCALE_X, DATA_SCALE_Y)
        nodeSprite = SprNode(position.hadamard_product(scale), (NODE_SIZE, NODE_SIZE))
        new_nodes.append(nodeSprite)
    return new_nodes


def main(args):
    # nodeInputFile = ''
    # edgeInputFile = ''
    # scale_x = 1
    # scale_y = 1
    #
    # if args.nodes:
    #     nodeInputFile = args.nodes
    # if args.edges:
    #     edgeInputFile = args.edges
    # if args.scale_x:
    #     scale_x = int(float(args.scale_x))
    # if args.scale_y:
    #     scale_y = int(float(args.scale_y))
    #
    # if nodeInputFile == edgeInputFile == '':
    #     print("Must provide at least one input file")
    #     sys.exit(2)

    # Instantiate NavMap. Right now, this is just a rectangle.
    # Create groups to hold enemy sprites and all sprites
    # - enemies is used for collision detection and position updates
    # - all_sprites is used for rendering
    nodeSprites = pygame.sprite.Group()
    all_sprites = pygame.sprite.Group()

    # Create the screen object
    # The size is determined by the constant SCREEN_WIDTH and SCREEN_HEIGHT
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    # Create Initial Sprite objects that are in the system
    nodes = []
    edges = []
    file = 'C:/Users/tburns/Documents/GitHub/coopgraph/data/test_data.csv'
    points = read_2d_points_from_data_file(file)
    NavMap = create_a_walk(points)

    new_nodes = create_node_sprites(NavMap)
    # edges = create_edges_from_points(create_points_from_file("C:\Users\tburns\Downloads\NodesAndEdges\Nodes.csv"))
    # edges = create_edges_from_points(create_random_points(20))

    # nodeSprites.add(new_nodes)
    # all_sprites.add(new_nodes)
    for node in new_nodes:
        nodeSprites.add(node)
        all_sprites.add(node)

    # Setup the clock for a decent framerate
    clock = pygame.time.Clock()

    # Run until the user asks to quit
    running = True
    while running:
        # Look at every event in the queue
        for event in pygame.event.get():
            # Did the user hit a key?
            if event.type == KEYDOWN:
                # Was it the Escape key? If so, stop the loop.
                if event.key == K_ESCAPE:
                    running = False
            # Did the user click the window close button? If so, stop the loop.
            elif event.type == QUIT:
                running = False

        # Get the set of keys pressed and check for user input
        # pressed_keys = pygame.key.get_pressed()

        # Update positions
        nodeSprites.update()

        # Update the display
        pygame.display.flip()

        # Fill the screen with black
        screen.fill((135, 206, 250))

        # Draw all sprites
        for entity in all_sprites:
            screen.blit(entity.surf, entity.rect)

        # Draw all lines
        for edge in NavMap.edges():
            scale = Vector2(DATA_SCALE_X, DATA_SCALE_Y)
            start = edge.start.pos.hadamard_product(scale)
            end = edge.end.pos.hadamard_product(scale)
            pygame.draw.line(screen, RED, (start.x, start.y), (end.x, end.y))

        # position_points = list(x._position for x in NavMap.nodes.values())
        # pygame.draw.aalines(screen, pygame.Color(255, 0, 0) , False, position_points)

        clock.tick(30)

    # Done! Time to quit.
    pygame.quit()


if __name__ == "__main__":
    import argparse

    text = 'NetworkMapViewer.py -nodes <nodefile> -edges <edgefile>'

    parser = argparse.ArgumentParser(description=text)
    parser.add_argument("-N", "--nodes", help="file path to .csv with nodes")
    parser.add_argument("-E", "--edges", help="file path to .csv with edges")
    parser.add_argument("-SY", "--scale_y", help="multiplier for all point coordinates in the y direction")
    parser.add_argument("-SX", "--scale_x", help="multiplier for all point coordinates in the x direction")
    args = parser.parse_args()

    main(args)
