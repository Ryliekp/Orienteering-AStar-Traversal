import sys
from dataclasses import dataclass
from enum import Enum
from PIL import Image
from math import *
from queue import PriorityQueue
from heapq import *


class Env(Enum):  # Color association
    OPEN_LAND = 5  # F89412 (248,148,18)
    ROUGH_MEADOW = 2.5  # FFC000 (255,192,0)
    EASY_MOVEMENT_FOREST = 3.75  # FFFFFF (255,255,255)
    SLOW_RUN_FOREST = 3.5  # 02D03C (2,208,60)
    WALK_FOREST = 3  # 028828 (2,136,40)
    IMPASSIBLE_VEGETATION = 0.2  # 054918 (5,73,24)
    LAKE_SWAMP_MARSH = 1  # 0000FF (0,0,255)
    PAVED_ROAD = 5.5  # 473303 (71,51,3)
    FOOTPATH = 4.5  # 000000 (0,0,0)
    OUT_OF_BOUNDS = 0  # CD0065 (205,0,101)


@dataclass
class Square:
    biome: Env
    coords: tuple
    elevation: float
    color: tuple
    g: float
    h: float
    f: float

    def __eq__(self, other):
        return self.coords == other.coords

    def __lt__(self, other):
        return self.g > other.g

    def __gt__(self, other):
        return self.g < other.g


# def heuristic(curr, end_node):
#     if curr.coords == end_node.coords:
#         return 0
#     dx = abs(curr.coords[0] - end_node.coords[0]) * 10.29
#     dy = abs(curr.coords[1] - end_node.coords[1]) * 7.55
#     ev = abs(end_node.elevation - curr.elevation)
#     hypotenuse = sqrt((dx ** 2) + (dy ** 2))
#     hypotenuse_with_ev = sqrt((hypotenuse ** 2) + (ev ** 2))
#     avg_speed = (curr.biome.value + end_node.biome.value) / 2
#     cost = hypotenuse_with_ev / avg_speed
#     return cost


# def estimatedCost(current, end_node):
#     cost = current[1] + heuristic(current[0], end_node)
#     return cost



# def childrenHelpFunc(new_node, popped, add_dist):
#     # 5262.467124743471
#     ev_diff = abs(popped[1][0].elevation - new_node.elevation)
#     distance = sqrt(ev_diff**2 + add_dist**2)
#     new_dist = popped[1][2] + distance
#     avg_speed = (new_node.biome.value + popped[1][0].biome.value) / 2
#     new_g = popped[1][2] + (distance / avg_speed)
#     new_tup = (new_node, new_g, new_dist)
#     # 5256.482322875155
#     # new_dist = popped[1][2] + add_dist
#     # avg_speed = (new_node.biome.value + popped[1][0].biome.value)/2
#     # new_g = popped[1][2] + (sqrt((new_node.elevation ** 2) + (new_dist ** 2))) / avg_speed
#     # new_tup = (new_node, new_g, new_dist)
#     return new_tup
#
#
# def getChildren(popped, array, end_node):  # popped = (cost, (node, g(n), distance))
#     x_coord, y_coord = popped[1][0].coords[0], popped[1][0].coords[1]
#     vert_dist = 7.55
#     horizontal_dist = 10.29
#     diagonal_dist = sqrt((7.55 ** 2) + (10.29 ** 2))
#     children = []
#
#     # West
#     if (x_coord - 1) >= 0:
#         new_node = array[y_coord][x_coord - 1]
#         if new_node.biome != Env.OUT_OF_BOUNDS:
#             new_tup = childrenHelpFunc(new_node, popped, horizontal_dist)
#             children.append((estimatedCost(new_tup, end_node), new_tup))
#         # North-West
#         if (y_coord - 1) >= 0:
#             new_node = array[y_coord - 1][x_coord - 1]
#             if new_node.biome != Env.OUT_OF_BOUNDS:
#                 new_tup = childrenHelpFunc(new_node, popped, diagonal_dist)
#                 children.append((estimatedCost(new_tup, end_node), new_tup))
#         # South-West
#         if (y_coord + 1) <= 499:
#             new_node = array[y_coord + 1][x_coord - 1]
#             if new_node.biome != Env.OUT_OF_BOUNDS:
#                 new_tup = childrenHelpFunc(new_node, popped, diagonal_dist)
#                 children.append((estimatedCost(new_tup, end_node), new_tup))
#     # East
#     if (x_coord + 1) <= 394:
#         new_node = array[y_coord][x_coord + 1]
#         if new_node.biome != Env.OUT_OF_BOUNDS:
#             new_tup = childrenHelpFunc(new_node, popped, horizontal_dist)
#             children.append((estimatedCost(new_tup, end_node), new_tup))
#         # North-East
#         if (y_coord - 1) >= 0:
#             new_node = array[y_coord - 1][x_coord + 1]
#             if new_node.biome != Env.OUT_OF_BOUNDS:
#                 new_tup = childrenHelpFunc(new_node, popped, diagonal_dist)
#                 children.append((estimatedCost(new_tup, end_node), new_tup))
#         # South-East
#         if (y_coord + 1) <= 499:
#             new_node = array[y_coord + 1][x_coord + 1]
#             if new_node.biome != Env.OUT_OF_BOUNDS:
#                 new_tup = childrenHelpFunc(new_node, popped, diagonal_dist)
#                 children.append((estimatedCost(new_tup, end_node), new_tup))
#     # North
#     if (y_coord - 1) >= 0:
#         new_node = array[y_coord - 1][x_coord]
#         if new_node.biome != Env.OUT_OF_BOUNDS:
#             new_tup = childrenHelpFunc(new_node, popped, vert_dist)
#             children.append((estimatedCost(new_tup, end_node), new_tup))
#     # South
#     if (y_coord + 1) <= 499:
#         new_node = array[y_coord + 1][x_coord]
#         if new_node.biome != Env.OUT_OF_BOUNDS:
#             new_tup = childrenHelpFunc(new_node, popped, vert_dist)
#             children.append((estimatedCost(new_tup, end_node), new_tup))
#     return children


def getDist(curr, old):
    add_dist = sqrt(((old.coords[1] - curr.coords[1]) * 7.55) ** 2 +
                    ((old.coords[0] - curr.coords[0]) * 10.29) ** 2)
    ev_diff = abs(old.elevation - curr.elevation)
    distance = sqrt(ev_diff ** 2 + add_dist ** 2)
    return distance


def getPath(start_node, end_node, parent_path, terrain_img, grid):
    """
    get the path from the start word to goal word found with bfs
    :param start_node: the starting pixel
    :param end_node: the goal pixel
    :param parent_path: a dictionary of pixels and their parents
    :param terrain_img: the image to draw the path on
    :return: a list of the steps in the path
    """
    path = [end_node.coords]
    while path[-1] != start_node.coords:
        try:
            path.append(parent_path[path[-1]])
        except KeyError:
            print("No solution")
            exit()
    path.reverse()

    dist = 0
    # px = img.load()
    for i in range(len(path)):
        pixel = path[i]
        new_dist = 0
        if(i > 0):
            prev_pix = path[i - 1]
            curr = grid[pixel[1]][pixel[0]]
            prev = grid[prev_pix[1]][prev_pix[0]]
            new_dist = getDist(curr, prev)
        dist += new_dist
        terrain_img.putpixel((pixel[0], pixel[1]), (118, 63, 231))
    return [terrain_img, dist]


# def aStar(start_node, end_node, array):
#     frontier = PriorityQueue()
#     current = (start_node, 0, 0)  # (node, g(n), distance)
#     cost = estimatedCost(current, end_node)
#     frontier.put((cost, current))
#     reached = {start_node.coords: current[1]}
#     parent_path = {}
#     while frontier:
#         popped = frontier.get()
#         if popped[1][0].coords == end_node.coords:
#             return [popped[1][2], parent_path]
#         for child in getChildren(popped, array, end_node):
#             child_node = child[1][0]
#             if child_node.coords not in reached:
#                 # If it is on the open list already,
#                 # check to see if this path to that square is better,
#                 # using G cost as the measure. A lower G cost means that
#                 # this is a better path. If so, change the parent of the
#                 # square to the current square, and recalculate the G and
#                 # F scores of the square. If you are keeping your open list
#                 # sorted by F score, you may need to resort the list to
#                 # account for the change.
#                 x = None
#                 for item in frontier.queue:
#                     if item[1][0].coords == child_node.coords:
#                         x = item
#                         break
#                 if x != None:
#                     if x[1][1] < child[1][1]:
#                         parent_path[x[1][0].coords] = popped[1][0].coords
#                         c = estimatedCost(child[1], end_node)
#                         frontier.put((c, child[1]))
#
#
#                 else:
#                     reached[child_node.coords] = child[0]
#                     frontier.put(child)
#                     if child_node.coords not in parent_path or child[0] < reached[child_node.coords]:
#                         parent_path[child_node.coords] = popped[1][0].coords
#     raise Exception("No Valid Path")

def getChildren2(popped, array, end_node):  # popped = (cost, (node, g(n), distance))
    x_coord, y_coord = popped.coords[0], popped.coords[1]
    vert_dist = 7.55
    horizontal_dist = 10.29
    diagonal_dist = sqrt((7.55 ** 2) + (10.29 ** 2))
    children = []

    # West
    if (x_coord - 1) >= 0:
        new_node = array[y_coord][x_coord - 1]
        if new_node.biome != Env.OUT_OF_BOUNDS:
            children.append(new_node)
        # North-West
        if (y_coord - 1) >= 0:
            new_node = array[y_coord - 1][x_coord - 1]
            if new_node.biome != Env.OUT_OF_BOUNDS:
                children.append(new_node)
        # South-West
        if (y_coord + 1) <= 499:
            new_node = array[y_coord + 1][x_coord - 1]
            if new_node.biome != Env.OUT_OF_BOUNDS:
                children.append(new_node)
    # East
    if (x_coord + 1) <= 394:
        new_node = array[y_coord][x_coord + 1]
        if new_node.biome != Env.OUT_OF_BOUNDS:
            children.append(new_node)
        # North-East
        if (y_coord - 1) >= 0:
            new_node = array[y_coord - 1][x_coord + 1]
            if new_node.biome != Env.OUT_OF_BOUNDS:
                children.append(new_node)
        # South-East
        if (y_coord + 1) <= 499:
            new_node = array[y_coord + 1][x_coord + 1]
            if new_node.biome != Env.OUT_OF_BOUNDS:
                children.append(new_node)
    # North
    if (y_coord - 1) >= 0:
        new_node = array[y_coord - 1][x_coord]
        if new_node.biome != Env.OUT_OF_BOUNDS:
            children.append(new_node)
    # South
    if (y_coord + 1) <= 499:
        new_node = array[y_coord + 1][x_coord]
        if new_node.biome != Env.OUT_OF_BOUNDS:
            children.append(new_node)
    return children

def getG(new_node, popped):
    add_dist = sqrt(((popped.coords[1] - new_node.coords[1])*7.55)**2 +
                    ((popped.coords[0] - new_node.coords[0])*10.29)**2)
    ev_diff = abs(popped.elevation - new_node.elevation)
    distance = sqrt(ev_diff ** 2 + add_dist ** 2)
    avg_speed = (new_node.biome.value + popped.biome.value) / 2
    new_g = (distance / avg_speed)
    return [new_g, distance]


def heuristic2(curr, end_node):
    if curr.coords == end_node.coords:
        return 0
    dx = abs(curr.coords[0] - end_node.coords[0]) * 10.29
    dy = abs(curr.coords[1] - end_node.coords[1]) * 7.55
    ev = abs(end_node.elevation - curr.elevation)
    hypotenuse = sqrt((dx ** 2) + (dy ** 2))
    hypotenuse_with_ev = sqrt((hypotenuse ** 2) + (ev ** 2))
    avg_speed = (curr.biome.value + end_node.biome.value) / 2
    cost = hypotenuse_with_ev / avg_speed
    return cost


def aAgain(start, end, img):
    frontier = [(start.f, start)]
    heapify(frontier)
    visited = {}
    parent = {}
    while frontier:
        curr = heappop(frontier)[1]
        visited[curr.coords] = curr
        if curr.coords == end.coords:
            return parent
        children = getChildren2(curr, img, end)
        for child in children:
            if child.coords not in visited:
                get_stuff = getG(child, curr)
                child.g = curr.g + get_stuff[0]
                child.h = heuristic2(child, end)
                child.f = child.g + child.h
                list = []
                x = -1
                length = len(frontier)
                for i in range(length):
                    pop = heappop(frontier)
                    pop_node = pop[1]
                    list.append(pop_node)
                    if pop_node.coords == child.coords:
                        x = i
                if x != -1:
                    if child.g <= list[x].g:
                        list.pop(x)
                        heappush(frontier, (child.f, child))
                        # parent[child.coords] = curr.coords
                        for item in list:
                            heappush(frontier, (item.f, item))
                        continue
                for item in list:
                    heappush(frontier, (item.f, item))
                heappush(frontier, (child.f, child))
                parent[child.coords] = curr.coords


def processImage(image, ev_file):
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
        print(processed_img[327][230])
        distance = 0
        for i in range(len(coords) - 1):
            start = processed_img[coords[i][1]][coords[i][0]]
            end = processed_img[coords[i + 1][1]][coords[i + 1][0]]
            # result = aStar(start, end, processed_img)
            result2 = aAgain(start, end, processed_img)
            paths = result2
            path_result = getPath(start, end, paths, img, processed_img)
            img = path_result[0]
            distance += path_result[1]
        print(distance)
        img.save(new_filename)
