"""
File: traversal.py
Author: Rylie Platt

Traversing a map to find the best path between given coordinates.

Takes 4 command line arguments map, elevations, destinations, map output

ex: python3 traversal.py, terrain.png, mpp.txt, red.txt, redOut.png
"""
import sys
from dataclasses import dataclass
from enum import Enum
from PIL import Image
from math import *
from heapq import *


class Env(Enum):                    # Color association
    """
    Environment Enum containing each environment and its associated travel speed
    """
    OPEN_LAND = 4.5                 # F89412 (248,148,18)
    ROUGH_MEADOW = 2.5              # FFC000 (255,192,0)
    EASY_MOVEMENT_FOREST = 3.75     # FFFFFF (255,255,255)
    SLOW_RUN_FOREST = 3.5           # 02D03C (2,208,60)
    WALK_FOREST = 3                 # 028828 (2,136,40)
    IMPASSIBLE_VEGETATION = 0.2     # 054918 (5,73,24)
    LAKE_SWAMP_MARSH = 0.5          # 0000FF (0,0,255)
    PAVED_ROAD = 5.5                # 473303 (71,51,3)
    FOOTPATH = 5                    # 000000 (0,0,0)
    OUT_OF_BOUNDS = 0               # CD0065 (205,0,101)


@dataclass
class Square:
    """
    dataclass representing a location on the map
    """
    biome: Env          # The Environment/speed of a cell
    coords: tuple       # The coords of a cell
    elevation: float    # the elevation of a cell
    color: tuple        # the rgb associated with the cell on a picture
    g: float            # the g value of a cell
    h: float            # the h value of a cell
    f: float            # the f value of a cell

    def __eq__(self, other):
        return self.coords == other.coords

    def __lt__(self, other):
        return self.g > other.g

    def __gt__(self, other):
        return self.g < other.g


def getDist(curr, old):
    """
    get the distance between the current and old cells
    :param curr: current cell
    :param old: old cell
    :return: distance between cells
    """
    add_dist = sqrt(((old.coords[1] - curr.coords[1]) * 7.55) ** 2 +
                    ((old.coords[0] - curr.coords[0]) * 10.29) ** 2)
    ev_diff = abs(old.elevation - curr.elevation)
    new_dist = sqrt(ev_diff ** 2 + add_dist ** 2)
    return new_dist


def getPath(start_cell, end_cell, parent_path, terrain_img, grid):
    """
    get the path from the start word to goal word found with bfs and
    draw it on the map
    :param grid: the 2D array of Square class instances
    :param start_cell: the starting pixel
    :param end_cell: the goal pixel
    :param parent_path: a dictionary of pixels and their parents
    :param terrain_img: the image to draw the path on
    :return: [the output image, the distance traveled]
    """
    path = [end_cell.coords]
    while path[-1] != start_cell.coords:
        try:
            path.append(parent_path[path[-1]])
        except KeyError:
            print("No solution")
            exit()
    path.reverse()

    total_dist = 0
    for step in range(len(path)):
        pixel = path[step]
        new_dist = 0
        if step > 0:
            prev_pix = path[step - 1]
            curr = grid[pixel[1]][pixel[0]]
            prev = grid[prev_pix[1]][prev_pix[0]]
            new_dist = getDist(curr, prev)
        total_dist += new_dist
        terrain_img.putpixel((pixel[0], pixel[1]), (118, 63, 231))
    return [terrain_img, total_dist]


def getChildren(popped, array):
    """
    get the children of the popped cell
    :param popped: the current cell
    :param array: 2D array of all cells on the map
    :return: an array of children
    """
    x_coord, y_coord = popped.coords[0], popped.coords[1]
    children = []

    # West
    if (x_coord - 1) >= 0:
        new_cell = array[y_coord][x_coord - 1]
        if new_cell.biome != Env.OUT_OF_BOUNDS:
            children.append(new_cell)
        # North-West
        if (y_coord - 1) >= 0:
            new_cell = array[y_coord - 1][x_coord - 1]
            if new_cell.biome != Env.OUT_OF_BOUNDS:
                children.append(new_cell)
        # South-West
        if (y_coord + 1) <= 499:
            new_cell = array[y_coord + 1][x_coord - 1]
            if new_cell.biome != Env.OUT_OF_BOUNDS:
                children.append(new_cell)
    # East
    if (x_coord + 1) <= 394:
        new_cell = array[y_coord][x_coord + 1]
        if new_cell.biome != Env.OUT_OF_BOUNDS:
            children.append(new_cell)
        # North-East
        if (y_coord - 1) >= 0:
            new_cell = array[y_coord - 1][x_coord + 1]
            if new_cell.biome != Env.OUT_OF_BOUNDS:
                children.append(new_cell)
        # South-East
        if (y_coord + 1) <= 499:
            new_cell = array[y_coord + 1][x_coord + 1]
            if new_cell.biome != Env.OUT_OF_BOUNDS:
                children.append(new_cell)
    # North
    if (y_coord - 1) >= 0:
        new_cell = array[y_coord - 1][x_coord]
        if new_cell.biome != Env.OUT_OF_BOUNDS:
            children.append(new_cell)
    # South
    if (y_coord + 1) <= 499:
        new_cell = array[y_coord + 1][x_coord]
        if new_cell.biome != Env.OUT_OF_BOUNDS:
            children.append(new_cell)
    return children


def getG(new_cell, popped):
    """
    get the g value from the current cell to a new cell
    :param new_cell: new child cell
    :param popped: current cell
    :return: the g value for the space between the given cells
    """
    add_dist = sqrt(((popped.coords[1] - new_cell.coords[1])*7.55)**2 +
                    ((popped.coords[0] - new_cell.coords[0])*10.29)**2)
    ev_diff = abs(popped.elevation - new_cell.elevation)
    distance = sqrt(ev_diff ** 2 + add_dist ** 2)
    avg_speed = (new_cell.biome.value + popped.biome.value) / 2
    new_g = (distance / avg_speed)
    return [new_g, distance]


def heuristic(curr, end_cell):
    """
    get the heuristic value between the current and goal cell
    :param curr: current cell
    :param end_cell: goal cell
    :return: the heuristic value between the current and goal cell
    """
    if curr.coords == end_cell.coords:
        return 0
    ev = abs(end_cell.elevation - curr.elevation)
    hypotenuse = getDist(curr, end_cell)
    hypotenuse_with_ev = sqrt((hypotenuse ** 2) + (ev ** 2))
    avg_speed = (curr.biome.value + end_cell.biome.value) / 2
    cost = hypotenuse_with_ev / avg_speed
    return cost


def aStar(start_cell, end_cell, image):
    """
    A-Star search to find shortest path from a start cell to an end cell
    :param start_cell: starting cell
    :param end_cell: end/goal cell
    :param image: map image
    :return: parent dictionary
    """
    frontier = [(start_cell.f, start_cell)]
    heapify(frontier)
    visited = {}
    parent = {}
    while frontier:
        curr = heappop(frontier)[1]
        visited[curr.coords] = curr
        if curr.coords == end_cell.coords:
            return parent
        children = getChildren(curr, image)
        for child in children:
            if child.coords not in visited:
                get_stuff = getG(child, curr)
                temp_g = curr.g + get_stuff[0]
                temp_h = heuristic(child, end_cell)
                temp_f = temp_g + temp_h
                if (child.f, child) in frontier:
                    list = []
                    x = -1
                    length = len(frontier)
                    for i in range(length):
                        pop = heappop(frontier)
                        pop_cell = pop[1]
                        list.append(pop_cell)
                        if pop_cell.coords == child.coords:
                            x = i
                            break
                    if temp_g < list[x].g:
                        list.pop(x)
                        child.g = temp_g
                        child.h = temp_h
                        child.f = temp_f
                        heappush(frontier, (child.f, child))
                        parent[child.coords] = curr.coords
                    for item in list:
                        heappush(frontier, (item.f, item))
                    continue
                child.g = temp_g
                child.h = temp_h
                child.f = temp_f
                heappush(frontier, (child.f, child))
                parent[child.coords] = curr.coords


def processImage(image, ev_file):
    """
    process image and create 2D array of Square instances corresponding to pixels
    :param image: the map image
    :param ev_file: the elevation file
    :return: a 2D array Square instances corresponding to pixels
    """
    pixels = image.load()
    file_lines = [line.rstrip('\n') for line in ev_file]
    img_width, img_height = image.size
    pix_array = []
    for row in range(img_height):  # row
        elevations = file_lines[row].split()[:395]
        pix_array.append([])
        for j in range(img_width):  # col
            pix_color = pixels[j, row][:3]
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
            pix_array[row].append(Square(pix_biome, (j, row), float(elevations[j]), pix_color, 0, 0, 0))
    return pix_array


if __name__ == '__main__':
    args = sys.argv
    with Image.open(args[1]) as img, open(args[2]) as file, open(args[3]) as destinations:
        new_filename = args[4]
        lines = [line.rstrip('\n') for line in destinations]
        coords = []
        for line in lines:
            x = line.split()
            coords.append((int(x[0]), int(x[1])))

        processed_img = processImage(img, file)
        distance = 0
        for i in range(len(coords) - 1):
            start = processed_img[coords[i][1]][coords[i][0]]
            end = processed_img[coords[i + 1][1]][coords[i + 1][0]]
            result = aStar(start, end, processed_img)
            paths = result
            path_result = getPath(start, end, paths, img, processed_img)
            img = path_result[0]
            distance += path_result[1]
        print(distance)
        img.save(new_filename)
