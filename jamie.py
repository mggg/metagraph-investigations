import time
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image, ImageDraw
from copy import deepcopy

class Direction:
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

def direction_from_int(direction):
    return ["RIGHT", "UP", "LEFT", "DOWN"][direction]

def turn_left(direction):
    return (direction + 1) % 4

def turn_right(direction):
    return (direction - 1) % 4

def get_cell_in_direction(board, row, column, direction):
    if direction == Direction.RIGHT:
        column += 1
    elif direction == Direction.UP:
        row -= 1
    elif direction == Direction.LEFT:
        column -= 1
    elif direction == Direction.DOWN:
        row += 1
    if row < 0 or column < 0 or row >= len(board) or column >= len(board[0]):
        return None, None, None
    else:
        return row, column, board[row][column]

def dfs(start_cell, board, reachable, district):
    start_row, start_column = start_cell
    if start_cell in reachable or board[start_row][start_column] != district:
        return
    reachable.add(start_cell)
    for diff_row, diff_column in [[0, 1], [-1, 0], [0, -1], [1, 0]]:
        row = start_row + diff_row
        column = start_column + diff_column
        new_cell = (row, column)
        if not (row < 0 or column < 0 or row >= len(board) or column >= len(board[0])):
            dfs(new_cell, board, reachable, district)

def is_head(board, bite_row, bite_column):
    district = board[bite_row][bite_column]
    reachable = {(bite_row, bite_column)}
    found_body = False
    for diff_row, diff_column in [[0, 1], [-1, 0], [0, -1], [1, 0]]:
        row = bite_row + diff_row
        column = bite_column + diff_column
        if (not (row < 0 or column < 0 or row >= len(board) or column >= len(board[0]))) and board[row][column] == district:
            start_cell = (row, column)
            if found_body and not (start_cell in reachable):
                    return False
            else:
                found_body = True
                dfs(start_cell, board, reachable, district)
    return True

def bite_left(board, bite_row, bite_column, direction):
    bite_row, bite_column, current_district = get_cell_in_direction(board, bite_row, bite_column, direction)
    direction = turn_left(direction)
    timeout = 0
    while True:
        timeout += 1
        if timeout >= 100:
            timeout -= 1
        new_row, new_column, new_district = get_cell_in_direction(board, bite_row, bite_column, direction)
        if new_row is None: # At edge of board.
            direction = turn_right(direction)
        elif new_district == current_district: # Can travel further down along district.
            bite_row = new_row
            bite_column = new_column
            direction = turn_left(direction)
        elif is_head(board, bite_row, bite_column): # Can bite.
            break
        else: # Can't bite, need to turn and then keep going.
            direction = turn_right(direction)
    board[bite_row][bite_column] = new_district
    return bite_row, bite_column, direction

def board_to_str(board):
    s = ""
    for row in board:
        for cell in row:
            s += str(cell)
        s += "\n"
    return s

def iteratively_print_biting_left(board, bite_row, bite_column, direction, wait_time, num_iterations):
    print(board_to_str(board))
    for i in range(num_iterations):
        time.sleep(wait_time)
        bite_row, bite_column, direction = bite_left(board, bite_row, bite_column, direction)
        print(board_to_str(board))

def iterate_biting_left(board, bite_row, bite_column, direction, num_iterations):
    for i in range(num_iterations):
        bite_row, bite_column, direction = bite_left(board, bite_row, bite_column, direction)
    return bite_row, bite_column, direction

def iteratively_draw_biting_left(board, bite_row, bite_column, direction, wait_time, num_iterations):
    im = plt.imshow(board, interpolation="nearest")
    plt.axis('off')
    plt.draw()
    for i in range(num_iterations):
        bite_row, bite_column, direction = bite_left(board, bite_row, bite_column, direction)
        if wait_time:
            plt.pause(wait_time)
        im.set_data(board)
    plt.show()

def find_cycle(board, bite_row, bite_column, direction, max_num_iterations):
    visited = {tuple(map(tuple, board)): 0}
    for i in range(max_num_iterations):
        bite_row, bite_column, direction = bite_left(board, bite_row, bite_column, direction)
        new_state = tuple(map(tuple, board))
        if new_state in visited:
            return board, i - visited[new_state] + 1, bite_row, bite_column, direction
        else:
            visited[new_state] = i
    return None, None, None, None, None

def find_and_print_cycle(board, bite_row, bite_column, direction, max_num_iterations):
    board, cycle_length, bite_row, bite_column, direction = find_cycle(board, bite_row, bite_column, direction, max_num_iterations)
    if board is None:
        print("No cycle found.")
    else:
        print(f"Cycle of length {cycle_length} found from initial state:")
        print(board_to_str(board) + f"Initial bite: {bite_row}, {bite_column}, {direction_from_int(direction)}")

def spiral_inward_right(num_rows, num_columns, district_size, add_one=True):
    district = 1
    board = [[0 for column in range(num_columns)] for row in range(num_rows)]
    if add_one:
        row = 0
        column = 0
        direction = Direction.RIGHT
        board[0][0] = 1
    else:
        row = 0
        column = -1
        direction = Direction.RIGHT
    while True:
        for i in range(district_size):
            for already_turned in [False, True]:
                new_row, new_column, new_district = get_cell_in_direction(board, row, column, direction)
                if (new_row is None) or new_district > 0:
                    if already_turned:
                        return board
                    else:
                        direction = turn_right(direction)
                else:
                    row = new_row
                    column = new_column
                    board[row][column] = district
                    break
        district += 1
    return board

def block_board(block_rows, block_columns, grid_rows, grid_columns):
    district = 0
    board = [[0 for column in range(block_columns*grid_columns)] for row in range(block_rows*grid_rows)]
    for grid_row in range(grid_rows):
        for grid_column in range(grid_columns):
            district += 1
            for block_row in range(block_columns):
                for block_column in range(block_columns):
                    board[grid_row*block_rows + block_row][grid_column*block_columns + block_column] = district
    return board

def make_into_spiral(board, bite_row, bite_column, direction, wait_time, max_num_iterations):
    im = plt.imshow(board, interpolation="nearest")
    plt.axis('off')
    plt.draw()
    for i in range(max_num_iterations):
        bite_row, bite_column, direction = bite_left(board, bite_row, bite_column, direction)
        if wait_time:
            plt.pause(wait_time)
        im.set_data(board)
    plt.show()

def score_bite(board_1, board_map_1, board_2_renamed, board_map_2_renamed, head, direction):
    """Returns how good it would be to bite from head in the given direction, higher numbers better.
    0 = invalid.
    1 = head belongs in current_district and has a path to the base in the intersection of board_1 and board_2.
    2 = head belongs in neither district.
    3 = head belongs in new_district and has a path to the base in the intersection of board_1 and board_2."""

def morph_greedy(board_1, board_2, wait_time, max_num_iterations):
    """Assumes district IDs are an interval of integers starting at 1."""
    bipartite_district_correspondence_graph = nx.Graph()
    same_district_graph_1 = nx.Graph()
    same_district_graph_2 = nx.Graph()
    max_district = 0
    num_rows = len(board_1)
    num_columns = len(board_1[0])
    for row in range(num_rows):
        for column in range(num_columns):
            district_1 = board_1[row][column]
            district_2 = board_2[row][column]
            if row > 0 and district_1 == board_1[row - 1][column]:
                same_district_graph_1.add_edge((row - 1, column), (row, column))
            if column > 0 and district_1 == board_1[row][column - 1]:
                same_district_graph_1.add_edge((row, column - 1), (row, column))
            if row > 0 and district_2 == board_2[row - 1][column]:
                same_district_graph_2.add_edge((row - 1, column), (row, column))
            if column > 0 and district_2 == board_2[row][column - 1]:
                same_district_graph_2.add_edge((row, column - 1), (row, column))
            if district_1 > max_district:
                max_district = district_1
            bipartite_district_correspondence_graph.add_edge(district_1, -district_2, base=(row, column))
    matching = nx.algorithms.matching.max_weight_matching(bipartite_district_correspondence_graph)

    board_2_renamed = [[0 for column in range(len(board_1[0]))] for row in range(len(board_1))]
    rename = {}
    base_of = list(range(max_district + 1))
    board_map_1 = [set() for _ in range(max_district + 1)]
    board_map_2_renamed = [set() for _ in range(max_district + 1)]
    for e in matching:
        district_1, district_2 = e
        if district_1 < 0:
            temp = -district_1
            district_1 = district_2
            district_2 = temp
        else:
            district_2 = -district_2
        base_of[district_1] = bipartite_district_correspondence_graph.get_edge_data(district_1, -district_2)["base"]
        rename[district_2] = district_1
    bases_set = set(base_of[1:])
    for row in range(num_rows):
        for column in range(num_columns):
            row_column = (row, column)
            district_1 = board_1[row][column]
            district_2_renamed = rename[board_2[row][column]]
            board_2_renamed[row][column] = district_2_renamed
            board_map_1[district_1].add(row_column)
            board_map_2_renamed[district_2_renamed].add(row_column)

    max_district_size = max(max(map(len, board_map_1[1:])), max(map(len, board_map_2_renamed[1:]))) + 1
    min_district_size = min(min(map(len, board_map_1[1:])), min(map(len, board_map_2_renamed[1:]))) - 1
    print(max_district_size)
    print(min_district_size)

    def reattach(row, column, new_district):
        same_district_graph_1.remove_node((row, column))
        if row > 0 and new_district == board_1[row - 1][column]:
            same_district_graph_1.add_edge((row - 1, column), (row, column))
        if column > 0 and new_district == board_1[row][column - 1]:
            same_district_graph_1.add_edge((row, column - 1), (row, column))
        if row < num_rows - 1 and new_district == board_1[row + 1][column]:
            same_district_graph_1.add_edge((row + 1, column), (row, column))
        if column < num_columns - 1 and new_district == board_1[row][column + 1]:
            same_district_graph_1.add_edge((row, column + 1), (row, column))

    def compute_loss():
        loss = [[max_district_size for column in range(len(board_1[0]))] for row in range(len(board_1))]
        same_district_graph_intersection = nx.intersection(same_district_graph_1, same_district_graph_2)
        #if same_district_graph_1.size() == same_district_graph_intersection.size():  # Partitions agree.
            #return 0
        frontier = set()
        for component in nx.connected_components(same_district_graph_intersection):
            num_bases = len(bases_set.intersection(component))
            if num_bases > 0:
                assert num_bases == 1, "Multiple bases in same component of intersection graph."
                for row, column in component:
                    loss[row][column] = 0
                    frontier.add((row, column))
        for distance in range(1, max_district_size):
            if len(frontier) == 0:
                break
            new_frontier = set()
            for vertex in frontier:
                for row, column in same_district_graph_1.neighbors(vertex):
                    if distance < loss[row][column]:
                        loss[row][column] = distance
                        new_frontier.add((row, column))
            frontier = new_frontier
        district_size_loss = 0
        for component in nx.connected_components(same_district_graph_1):
            length = len(component)
            if length == min_district_size or length == max_district_size:
                district_size_loss += 1
        return sum(map(sum, loss)) + 0.0001*district_size_loss

    im = plt.imshow(board_1, interpolation="nearest")
    plt.axis('off')
    plt.draw()
    plt.pause(.00001)

    initial_loss_sum = compute_loss()
    progress = "Error: Progress should have been set."
    for iteration in range(max_num_iterations + 1):
        print(initial_loss_sum)
        if initial_loss_sum == 0:
            progress = f"Successfully morphed districts after {iteration} iterations."
            break
        elif iteration == max_num_iterations:
            progress = f"Not enough iterations, final loss = {initial_loss_sum}."
            break
        min_loss = initial_loss_sum, None, None, None
        for row in range(num_rows):
            for column in range(num_columns):
                old_district = board_1[row][column]
                if len(board_map_1[old_district]) > min_district_size and is_head(board_1, row, column):
                    for diff_row, diff_column in [[0, 1], [-1, 0], [0, -1], [1, 0]]:
                        new_row = row + diff_row
                        new_column = column + diff_column
                        if new_row >= 0 and new_row < num_rows and new_column >= 0 and new_column < num_columns:
                            new_district = board_1[new_row][new_column]
                            if len(board_map_1[board_1[new_row][new_column]]) < max_district_size:
                                reattach(row, column, new_district)
                                new_loss_sum = compute_loss()
                                reattach(row, column, old_district)
                                if new_loss_sum < min_loss[0]:
                                    min_loss = new_loss_sum, row, column, new_district
        new_loss_sum, row, column, new_district = min_loss
        if new_loss_sum == initial_loss_sum:
            progress = f"Hit local minimum on iteration {iteration} at loss = {initial_loss_sum}."
            break
        else:
            initial_loss_sum = new_loss_sum
            board_map_1[board_1[row][column]].remove((row, column))
            board_map_1[new_district].add((row, column))
            board_1[row][column] = new_district
            reattach(row, column, new_district)
        if wait_time:
            plt.pause(wait_time)
        im.set_data(board_1)
    print(progress)
    im.set_data(board_1)
    plt.pause(.00001)
    plt.show()

initial_board_1 = [["a", "a", "a"],
                   ["b", "c", "d"],
                   ["b", "c", "d"]]

initial_board_2 = [["a", "a", "a", "b", "g", "g"],
                   ["a", "a", "c", "b", "g", "g"],
                   ["c", "c", "c", "b", "g", "f"],
                   ["c", "c", "b", "b", "f", "f"],
                   ["d", "e", "e", "e", "e", "f"],
                   ["d", "d", "d", "d", "e", "f"]]

initial_board_3 = [[1, 1, 1],
                   [2, 3, 4],
                   [2, 3, 4]]

initial_board_4 = [[1, 1, 1, 2, 7, 7],
                   [1, 1, 3, 2, 7, 7],
                   [3, 3, 3, 2, 7, 6],
                   [3, 3, 2, 2, 6, 6],
                   [4, 5, 5, 5, 5, 6],
                   [4, 4, 4, 4, 5, 6]]

initial_board_5 = [[1, 1, 1, 1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2, 2, 2, 2]]

initial_board_6 = [[4, 3, 3],
                   [4, 2, 1],
                   [2, 2, 1]]

initial_board_7 = [[1, 2, 3, 4, 5, 6],
                   [1, 2, 3, 4, 5, 6],
                   [1, 2, 3, 4, 5, 6],
                   [1, 2, 3, 4, 5, 6],
                   [1, 2, 3, 4, 5, 6],
                   [7, 7, 7, 7, 7, 7]]

initial_board_8 = [
    [11, 11, 11, 11, 11, 14, 57, 57, 21, 21, 21, 21, 21, 21, 23, 23, 23, 23, 23, 23, 7, 7, 78, 78, 78, 78, 78, 78, 78,
     78],
    [11, 14, 14, 14, 14, 14, 14, 57, 21, 21, 21, 20, 20, 20, 23, 4, 4, 4, 23, 23, 7, 7, 7, 78, 78, 53, 53, 13, 59, 78],
    [11, 14, 52, 14, 14, 14, 57, 57, 57, 21, 20, 20, 20, 20, 20, 45, 45, 4, 4, 23, 23, 7, 7, 39, 53, 53, 53, 13, 59,
     59],
    [11, 52, 52, 52, 52, 57, 57, 74, 57, 21, 20, 20, 20, 43, 45, 45, 4, 4, 4, 4, 7, 7, 39, 39, 53, 13, 13, 13, 59, 59],
    [11, 11, 11, 52, 52, 57, 57, 74, 74, 43, 43, 43, 43, 43, 45, 45, 12, 12, 4, 4, 7, 7, 39, 39, 53, 13, 13, 6, 6, 59],
    [56, 56, 56, 52, 52, 58, 74, 74, 5, 43, 5, 5, 43, 43, 43, 45, 45, 12, 12, 12, 39, 39, 39, 39, 53, 13, 13, 13, 6,
     59],
    [56, 58, 52, 52, 58, 58, 58, 74, 5, 5, 5, 77, 43, 64, 45, 45, 64, 12, 22, 22, 22, 22, 39, 53, 53, 70, 13, 6, 6, 59],
    [56, 58, 58, 58, 58, 58, 37, 74, 5, 5, 77, 77, 77, 64, 45, 64, 64, 12, 12, 22, 22, 22, 39, 46, 53, 70, 70, 6, 6,
     59],
    [56, 58, 37, 37, 37, 37, 37, 74, 5, 77, 77, 47, 47, 64, 64, 64, 64, 12, 22, 22, 46, 46, 46, 46, 70, 70, 40, 6, 6,
     59],
    [56, 56, 56, 37, 37, 74, 74, 74, 5, 77, 1, 9, 47, 64, 64, 12, 12, 12, 22, 46, 46, 49, 31, 70, 70, 70, 40, 40, 6,
     59],
    [56, 56, 37, 37, 37, 19, 19, 19, 5, 77, 1, 9, 47, 47, 47, 47, 47, 47, 22, 46, 49, 49, 31, 70, 70, 68, 40, 40, 6,
     73],
    [2, 2, 2, 2, 76, 19, 1, 77, 77, 77, 1, 9, 9, 9, 9, 47, 32, 47, 46, 46, 46, 49, 31, 70, 68, 68, 40, 40, 40, 73],
    [2, 2, 44, 2, 76, 19, 1, 1, 1, 1, 1, 9, 32, 32, 9, 32, 32, 72, 72, 72, 72, 49, 31, 31, 68, 68, 40, 73, 73, 73],
    [44, 2, 44, 2, 76, 19, 1, 1, 19, 79, 79, 9, 32, 32, 32, 32, 33, 24, 72, 49, 72, 49, 49, 31, 31, 68, 40, 40, 73, 73],
    [44, 44, 44, 2, 76, 19, 19, 19, 19, 79, 79, 9, 32, 32, 33, 33, 33, 24, 72, 49, 49, 49, 31, 31, 31, 68, 8, 8, 73,
     73],
    [44, 51, 51, 2, 76, 76, 30, 30, 30, 79, 79, 9, 33, 33, 33, 36, 24, 24, 72, 72, 16, 16, 31, 38, 68, 68, 68, 8, 73,
     60],
    [44, 44, 51, 51, 76, 76, 76, 30, 79, 79, 65, 65, 65, 33, 33, 36, 36, 24, 72, 16, 16, 16, 38, 38, 68, 38, 8, 8, 73,
     60],
    [44, 51, 51, 51, 63, 76, 76, 30, 30, 79, 65, 65, 65, 65, 33, 33, 36, 24, 72, 24, 16, 67, 38, 38, 38, 38, 38, 8, 60,
     60],
    [44, 51, 51, 71, 63, 63, 63, 63, 30, 79, 65, 62, 26, 26, 26, 36, 36, 24, 24, 24, 16, 67, 38, 38, 8, 8, 8, 8, 60,
     60],
    [51, 51, 71, 71, 63, 29, 63, 63, 30, 79, 65, 62, 26, 36, 36, 36, 36, 36, 75, 24, 16, 67, 67, 67, 67, 8, 60, 60, 60,
     60],
    [71, 71, 71, 63, 63, 29, 63, 29, 30, 65, 65, 62, 26, 26, 26, 26, 26, 75, 75, 16, 16, 16, 67, 67, 34, 34, 34, 34, 60,
     17],
    [50, 50, 71, 71, 71, 29, 29, 29, 30, 30, 62, 62, 62, 26, 75, 75, 75, 75, 75, 75, 48, 67, 67, 67, 34, 17, 17, 17, 17,
     17],
    [50, 50, 28, 18, 71, 71, 29, 29, 29, 29, 62, 3, 62, 26, 54, 75, 75, 55, 55, 48, 48, 48, 34, 34, 34, 27, 27, 17, 17,
     35],
    [50, 28, 28, 18, 18, 18, 18, 18, 18, 29, 3, 3, 62, 54, 54, 41, 41, 41, 55, 55, 55, 48, 48, 34, 34, 27, 27, 27, 17,
     35],
    [50, 66, 28, 28, 28, 28, 28, 18, 18, 18, 3, 62, 62, 61, 54, 41, 41, 41, 41, 41, 55, 55, 48, 48, 34, 15, 27, 17, 17,
     35],
    [50, 66, 66, 28, 10, 28, 28, 18, 3, 3, 3, 61, 61, 61, 54, 41, 25, 25, 25, 41, 55, 48, 48, 69, 15, 15, 27, 27, 35,
     35],
    [50, 50, 66, 66, 10, 10, 10, 3, 3, 42, 3, 3, 61, 61, 54, 41, 54, 25, 55, 55, 55, 25, 48, 69, 15, 27, 27, 27, 35,
     35],
    [50, 66, 66, 66, 10, 10, 10, 10, 10, 42, 42, 42, 42, 61, 54, 54, 54, 25, 25, 25, 25, 25, 69, 69, 15, 15, 15, 15, 15,
     35],
    [50, 66, 66, 66, 10, 10, 42, 42, 42, 42, 42, 42, 61, 61, 61, 61, 54, 25, 69, 69, 69, 69, 69, 69, 69, 15, 15, 35, 35,
     35]]

initial_board_9 = [[15, 15, 15, 44, 44, 44, 44, 44, 33, 33, 33, 2, 2, 2, 2, 2, 2, 2, 30, 30, 30, 30, 30],
                   [41, 15, 15, 15, 15, 44, 44, 33, 33, 33, 2, 2, 2, 9, 9, 2, 9, 39, 30, 24, 24, 30, 30],
                   [41, 41, 23, 23, 15, 44, 44, 44, 33, 33, 33, 9, 9, 9, 9, 9, 9, 39, 30, 24, 24, 22, 22],
                   [41, 23, 23, 23, 15, 15, 44, 33, 33, 17, 17, 17, 17, 9, 9, 16, 39, 39, 30, 30, 24, 22, 22],
                   [41, 23, 41, 23, 23, 15, 32, 32, 32, 32, 32, 32, 17, 17, 17, 16, 39, 39, 39, 39, 24, 24, 22],
                   [41, 41, 41, 10, 23, 23, 13, 32, 32, 32, 17, 17, 17, 16, 16, 16, 16, 16, 39, 24, 24, 24, 22],
                   [41, 10, 41, 10, 10, 23, 13, 32, 25, 32, 4, 4, 17, 16, 16, 16, 16, 39, 39, 26, 22, 24, 22],
                   [10, 10, 10, 10, 10, 13, 13, 13, 25, 25, 4, 11, 11, 11, 11, 26, 26, 26, 26, 26, 22, 22, 22],
                   [10, 8, 8, 13, 10, 13, 25, 25, 25, 25, 4, 11, 45, 45, 11, 11, 11, 5, 26, 26, 26, 26, 26],
                   [6, 8, 8, 13, 13, 13, 25, 4, 25, 25, 4, 4, 45, 45, 11, 5, 5, 5, 5, 7, 47, 47, 47],
                   [6, 6, 8, 8, 3, 13, 25, 4, 4, 4, 4, 45, 45, 45, 11, 11, 42, 42, 5, 7, 7, 47, 47],
                   [6, 8, 8, 8, 3, 3, 3, 3, 3, 3, 3, 45, 45, 42, 42, 42, 42, 5, 5, 7, 7, 7, 47],
                   [6, 6, 8, 8, 3, 29, 29, 19, 3, 3, 21, 45, 42, 42, 42, 1, 1, 5, 5, 5, 7, 7, 47],
                   [6, 6, 29, 29, 29, 29, 19, 19, 19, 19, 21, 45, 21, 42, 42, 1, 1, 1, 1, 1, 1, 7, 47],
                   [6, 6, 6, 29, 19, 19, 19, 20, 20, 21, 21, 21, 21, 21, 1, 1, 18, 1, 27, 1, 7, 7, 47],
                   [40, 40, 40, 29, 29, 19, 19, 20, 20, 20, 36, 36, 21, 21, 18, 18, 18, 18, 27, 27, 27, 47, 47],
                   [40, 40, 40, 40, 29, 19, 38, 38, 20, 20, 36, 36, 36, 21, 31, 31, 18, 18, 18, 18, 27, 27, 27],
                   [35, 40, 40, 40, 29, 38, 38, 20, 20, 20, 14, 36, 36, 31, 31, 31, 18, 18, 12, 12, 12, 27, 27],
                   [35, 35, 35, 40, 43, 38, 38, 38, 38, 20, 14, 36, 36, 37, 37, 31, 31, 31, 31, 31, 12, 12, 27],
                   [28, 28, 35, 35, 43, 43, 38, 14, 14, 14, 14, 37, 36, 36, 37, 31, 37, 12, 12, 12, 12, 12, 27],
                   [28, 35, 35, 43, 43, 43, 38, 38, 14, 48, 14, 37, 37, 37, 37, 37, 37, 12, 34, 34, 34, 34, 34],
                   [28, 28, 35, 35, 35, 43, 43, 14, 14, 48, 14, 48, 48, 46, 46, 46, 46, 34, 34, 34, 34, 34, 34],
                   [28, 28, 28, 28, 28, 28, 43, 43, 43, 48, 48, 48, 48, 48, 48, 48, 46, 46, 46, 46, 46, 46, 46]]

initial_board_10 = [[22, 22, 22, 22, 22, 11, 11, 11, 11, 11, 11, 2, 2, 2, 2, 2, 17, 17],
                    [22, 26, 26, 22, 11, 11, 33, 33, 33, 25, 11, 2, 2, 5, 5, 2, 2, 17],
                    [22, 26, 26, 16, 16, 16, 16, 16, 33, 25, 25, 25, 5, 5, 5, 5, 5, 17],
                    [22, 26, 26, 16, 24, 24, 16, 33, 33, 25, 25, 25, 25, 4, 5, 5, 5, 17],
                    [26, 26, 26, 16, 24, 34, 16, 33, 4, 25, 4, 4, 4, 4, 12, 12, 13, 17],
                    [21, 21, 21, 24, 24, 34, 33, 33, 4, 4, 4, 12, 12, 12, 12, 13, 13, 17],
                    [21, 21, 21, 36, 24, 34, 34, 34, 32, 32, 3, 12, 3, 3, 12, 13, 14, 17],
                    [21, 21, 21, 36, 24, 34, 32, 34, 32, 3, 3, 12, 3, 13, 13, 13, 14, 17],
                    [7, 7, 36, 36, 24, 34, 32, 32, 32, 31, 3, 3, 3, 35, 13, 13, 14, 14],
                    [7, 7, 36, 15, 24, 34, 8, 32, 32, 31, 35, 35, 35, 35, 14, 14, 14, 14],
                    [7, 36, 36, 15, 15, 15, 8, 8, 31, 31, 35, 23, 35, 35, 1, 14, 1, 19],
                    [7, 7, 36, 36, 15, 15, 8, 8, 8, 31, 35, 23, 1, 1, 1, 1, 1, 19],
                    [7, 28, 28, 28, 15, 8, 8, 8, 31, 31, 31, 23, 1, 6, 6, 19, 19, 19],
                    [7, 28, 28, 15, 15, 10, 10, 10, 31, 29, 23, 23, 23, 6, 6, 19, 19, 19],
                    [28, 28, 9, 10, 10, 10, 10, 29, 29, 29, 29, 23, 6, 6, 20, 19, 30, 30],
                    [28, 9, 9, 9, 9, 9, 10, 10, 29, 18, 18, 23, 6, 20, 20, 30, 30, 30],
                    [28, 27, 9, 9, 9, 27, 27, 29, 29, 29, 18, 23, 6, 20, 20, 30, 30, 30],
                    [27, 27, 27, 27, 27, 27, 18, 18, 18, 18, 18, 18, 6, 20, 20, 20, 20, 30]]

test_num = 24

if test_num == 1:
    iteratively_print_biting_left(initial_board_1, 1, 2, Direction.UP, .5, 25)
elif test_num == 2:
    iteratively_print_biting_left(initial_board_2, 3, 2, Direction.UP, .5, 100)
elif test_num == 3:
    find_and_print_cycle(initial_board_1, 1, 2, Direction.UP, 100)
elif test_num == 4:
    find_and_print_cycle(initial_board_2, 3, 2, Direction.UP, 100000)
elif test_num == 5:
    find_and_print_cycle(initial_board_2, 0, 1, Direction.DOWN, 100000)
elif test_num == 6:
    for i in range(5):
        A = np.random.rand(5, 8)
        print(A)
        x = (0,2)
        print(A[x])
        plt.imshow(A, interpolation="nearest")
        plt.draw()
        plt.pause(0.5)
    plt.show()
elif test_num == 7:
    plt.imshow(initial_board_4, interpolation="nearest")
    plt.show()
elif test_num == 8:
    iteratively_draw_biting_left(initial_board_4, 3, 2, Direction.UP, .1, 1000)
elif test_num == 9:
    iteratively_draw_biting_left(initial_board_5, 0, 0, Direction.DOWN, .1, 1000)
elif test_num == 10:
    print(board_to_str(spiral_inward_right(29, 30, 11)))
elif test_num == 11:
    board = spiral_inward_right(29, 30, 11)
    iteratively_draw_biting_left(board, 0, 0, Direction.DOWN, .1, 1000)
elif test_num == 12:
    board = spiral_inward_right(29, 30, 11)
    iteratively_draw_biting_left(board, 0, 0, Direction.DOWN, .00001, 100000)
elif test_num == 13:
    board = spiral_inward_right(29, 30, 11)
    iteratively_draw_biting_left(board, 0, 0, Direction.DOWN, 0, 100000)
elif test_num == 14:
    board = spiral_inward_right(29, 30, 11)
    iteratively_draw_biting_left(board, 10, 10, Direction.DOWN, .00001, 100000)
elif test_num == 15:
    board = spiral_inward_right(29, 30, 11)
    iteratively_draw_biting_left(board, 10, 10, Direction.DOWN, 0, 100000)
elif test_num == 16:
    morph_greedy(initial_board_3, initial_board_6, 1, 5)
elif test_num == 17:
    morph_greedy(initial_board_4, initial_board_7, 1, 100)
elif test_num == 18:
    board = spiral_inward_right(29, 30, 11)
    iterate_biting_left(board, 1, 0, Direction.UP, 100000)
    print(f"initial_board_8 = {board}")
elif test_num == 19:
    board_2 = spiral_inward_right(29, 30, 11)
    board_1 = initial_board_8
    morph_greedy(board_1, board_2, .00001, 10)
elif test_num == 20:
    board = spiral_inward_right(23, 23, 11)
    iterate_biting_left(board, 1, 0, Direction.UP, 100000)
    print(f"initial_board_9 = {board}")
elif test_num == 21:
    board_2 = spiral_inward_right(23, 23, 11)
    board_1 = initial_board_9
    morph_greedy(board_1, board_2, .00001, 900)
elif test_num == 22:
    board = spiral_inward_right(18, 18, 9, False)
    iterate_biting_left(board, 1, 0, Direction.UP, 100000)
    print(f"initial_board_10 = {board}")
elif test_num == 23:  # Success.
    board_1 = initial_board_10
    board_2 = block_board(3, 3, 6, 6)
    morph_greedy(board_1, board_2, .00001, 900)
elif test_num == 24:  # Success.
    board_1 = spiral_inward_right(18, 18, 9, False)
    board_2 = block_board(3, 3, 6, 6)
    morph_greedy(board_1, board_2, .00001, 900)