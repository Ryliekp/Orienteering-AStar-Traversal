import sys
from dataclasses import dataclass
from enum import Enum
from PIL import Image
from heapq import *
from math import *


class Env(Enum):                    # Color association
    OPEN_LAND = 4.5                 # F89412 (248,148,18)
    ROUGH_MEADOW = 3                # FFC000 (255,192,0)
    EASY_MOVEMENT_FOREST = 3.5      # FFFFFF (255,255,255)
    SLOW_RUN_FOREST = 3             # 02D03C (2,208,60)
    WALK_FOREST = 2.5               # 028828 (2,136,40)
    IMPASSIBLE_VEGETATION = 0.2     # 054918 (5,73,24)
    LAKE_SWAMP_MARSH = 1.2          # 0000FF (0,0,255)
    PAVED_ROAD = 5                  # 473303 (71,51,3)
    FOOTPATH = 3.5                  # 000000 (0,0,0)
    OUT_OF_BOUNDS = 0               # CD0065 (205,0,101)


@dataclass
class Square:
    biome: Env
    coords: tuple
    elevation: float
    color: tuple


def heuristic(curr, end):
    d = 1
    d2 = 2**(1/2)
    dx = abs(curr.coors[0] - end.coords[0]) * 10.29
    dy = abs(curr.coors[1] - end.coords[1]) * 7.55
    h_dist = d * (dx + dy) + (d2 - 2 * d) * min(dx, dy)
    ev = abs(end.elevation - curr.elevation)
    pythagorean = sqrt((h_dist**2) + (ev**2))
    cost = end.biome.value / pythagorean
    return cost


def estimatedCost(current, end):
    cost = current(1) + heuristic(current[0], end)
    return cost


def childrenHelpFunc(x, y, popped, add_dist, array):
    new_dist = popped[1][3] + add_dist
    new_node = array[y][x]
    new_g = popped[1][3] + (new_node.biome.value /
                            sqrt((new_node.elevation ** 2) + (new_dist ** 2)))
    new_tup = (new_node, new_g, new_dist)
    return new_tup


def getChildren(popped, array, end):   # popped = (cost, (node, g(n), distance))
    x, y = popped[1][0].coords[0], popped[1][0].coords[0][1]
    vert_dist = 7.55
    horiz_dist = 10.29
    diag_dist = sqrt((7.55**2) + (10.29**2))
    children = []
    # East
    if (x-1) >= 0:
        new_tup = childrenHelpFunc(x-1, y, popped, horiz_dist, array)
        if new_tup[0].biome != Env.OUT_OF_BOUNDS:
            children.append((estimatedCost(new_tup, end), new_tup))
        # North-East
        if (y-1) >= 0:
            new_tup = childrenHelpFunc(x - 1, y - 1, popped, diag_dist, array)
            if new_tup[0].biome != Env.OUT_OF_BOUNDS:
                children.append((estimatedCost(new_tup, end), new_tup))
        # South-East
        if (y+1) <= 499:
            new_tup = childrenHelpFunc(x - 1, y + 1, popped, diag_dist, array)
            if new_tup[0].biome != Env.OUT_OF_BOUNDS:
                children.append((estimatedCost(new_tup, end), new_tup))
    # West
    if (x+1) <= 394:
        new_tup = childrenHelpFunc(x + 1, y, popped, horiz_dist, array)
        if new_tup[0].biome != Env.OUT_OF_BOUNDS:
            children.append((estimatedCost(new_tup, end), new_tup))
        # North-West
        if (y - 1) >= 0:
            new_tup = childrenHelpFunc(x + 1, y - 1, popped, diag_dist, array)
            if new_tup[0].biome != Env.OUT_OF_BOUNDS:
                children.append((estimatedCost(new_tup, end), new_tup))
        # South-West
        if (y + 1) <= 499:
            new_tup = childrenHelpFunc(x + 1, y + 1, popped, diag_dist, array)
            if new_tup[0].biome != Env.OUT_OF_BOUNDS:
                children.append((estimatedCost(new_tup, end), new_tup))
    # North
    if (y-1) >= 0:
        new_tup = childrenHelpFunc(x, y + 1, popped, vert_dist, array)
        if new_tup[0].biome != Env.OUT_OF_BOUNDS:
            children.append((estimatedCost(new_tup, end), new_tup))
    # South
    if (y+1) <= 499:
        new_tup = childrenHelpFunc(x, y + 1, popped, vert_dist, array)
        if new_tup[0].biome != Env.OUT_OF_BOUNDS:
            children.append((estimatedCost(new_tup, end), new_tup))
    return 0


def aStar(start, end, paths, array):
    # function
    # BEST-FIRST-SEARCH(problem, f)
    # node←NODE(STATE=problem.INITIAL)
    # frontier←a priority queue ordered by f, with node as an element
    frontier = []
    current = (start, 0, 0)    # (node, g(n), distance)
    cost = estimatedCost(current, end)
    heappush(frontier, (cost, current))
    # reached←a lookup table, with one entry with key problem.INITIAL and value node
    reached = {start: cost}
    # while not IS-EMPTY(frontier) do
    while frontier:
        # node←POP(frontier)
        popped = heappop(frontier)     # popped = (cost, (node, g(n), distance))
        # if problem.IS - GOAL(node.STATE) then return node
        if popped[1][1] == end:
            return popped[1][2]
        # for each child in EXPAND(problem, node) do
        for child in getChildren(popped, array, end):
            x = 'bob'   # placeholder
        #   s←child.STATE
        #   if s is not in reached or child.PATH-COST < reached[s].PATH-COST then
        #       reached[s]←child
        #       add child to frontier
    # return failure


def processImage(image, ev_file):
    pixels = image.load()
    lines = [line.rstrip('\n') for line in ev_file]
    img_width, img_height = image.size
    pix_array = []
    for i in range(img_height):  # row
        elevations = lines[i].split()[:395]
        pix_array.append([])
        for j in range(img_width):  # col
            pix_color = pixels[j, i][:3]
            match pix_color:
                case (248, 148, 18):
                    pix_biome = Env.OPEN_LAND
                case (255, 192, 0):
                    pix_biome = Env.ROUGH_MEADOW
                case (255, 255, 255):
                    pix_biome = Env.EASY_MOVEMENT_FOREST
                case (2, 208, 60):
                    pix_biome = Env.SLOW_RUN_FOREST
                case (2, 136, 40):
                    pix_biome = Env.WALK_FOREST
                case (5, 73, 24):
                    pix_biome = Env.IMPASSIBLE_VEGETATION
                case (0, 0, 255):
                    pix_biome = Env.LAKE_SWAMP_MARSH
                case (71, 51, 3):
                    pix_biome = Env.PAVED_ROAD
                case (0, 0, 0):
                    pix_biome = Env.FOOTPATH
                case (205, 0, 101):
                    pix_biome = Env.OUT_OF_BOUNDS
                case _:
                    raise Exception("Unknown Color")
            pix_array[i].append(Square(pix_biome, (j, i), float(elevations[j]), pix_color))
    return pix_array


if __name__ == '__main__':
    # args = sys.argv
    args = [0, 'terrain.png', 'mpp.txt']
    with Image.open(args[1]) as img, open(args[2]) as file:
        processed_img = processImage(img, file)
        # distance = aStar(start, end, paths, processed_img)
        # print(processed_img)
