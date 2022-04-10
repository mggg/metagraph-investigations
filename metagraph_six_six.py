import networkx as nx
from tqdm import tqdm
from sys import argv


blocks = (((1, 0), (2, 0)), ((0, 1), (0, 2)), ((1, 0), (0, 1)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((1, 0), (1, -1)))


def try_to_add(p, district_number, cells):
    for row, col in cells:
        if row < 0 or row >= 6 or col < 0 or col >= 6:
            return False
        i = 6*row + col
        if p[i] == 0:
            p[i] = district_number
        else:
            return False
    return True


def canonize(p):
    s = {0}
    renaming = [0 for _ in range(13)]
    new_name = 0
    for old_name in p:
        if not(old_name in s):
            s.add(old_name)
            new_name += 1
            renaming[old_name] = new_name
    return tuple(renaming[old_name] for old_name in p)


def fill(root):
    stack = [root]
    partitions = []
    while len(stack) > 0:
        p_1, m = stack.pop()
        if m == 12:
            partitions.append(tuple(p_1))
        else:
            m += 1
            i = p_1.index(0)
            row_base = int(i / 6)
            col_base = i % 6
            for cell_2, cell_3 in blocks:
                p_2 = p_1.copy()
                if try_to_add(p_2, m, [(row_base, col_base), (row_base + cell_2[0], col_base + cell_2[1]), (row_base + cell_3[0], col_base + cell_3[1])]):
                    stack.append((p_2, m))
    return partitions


def print_nicely(p):
    for row in range(6):
        s = ""
        for col in range(6):
            s += chr(ord('@') + p[6*row + col])
        print(s)


if __name__ == "__main__":
    command = argv[1] if len(argv) > 1 else ""
    if command == "generate":
        partitions = fill(([0 for _ in range(36)], 0))
        print(f"There are {len(partitions)} partitions of a 6x6 grid graph into districts of size 3.")
        metagraph = nx.Graph()
        metagraph.add_nodes_from(partitions)
        for p_1 in tqdm(partitions):
            for name_1 in range(1, 13):
                for name_2 in range(1, 13):
                    if name_1 != name_2:
                        p_1_with_2_districts_removed = list(p_1)
                        for i in range(36):
                            x = p_1_with_2_districts_removed[i]
                            if x == name_1 or x == name_2:
                                p_1_with_2_districts_removed[i] = 0
                        neighbors = fill((list(canonize(p_1_with_2_districts_removed)), 10))
                        if len(neighbors) > 1:
                            canonical_neighbors = map(canonize, neighbors)
                            for p_2 in canonical_neighbors:
                                if p_1 != p_2:
                                    metagraph.add_edge(p_1, p_2)
        nx.write_gpickle(metagraph, "metagraph6x6.p")
    elif command == "analyze":
        metagraph = nx.read_gpickle("metagraph6x6.p")
        print(f"The metagraph has {len(metagraph.nodes())} vertices and {len(metagraph.edges())} edges.")
        components = list(nx.connected_components(metagraph))
        print(f"\nThere are {len(components)} connected components.")
        print("\nSizes of components:")
        component_sizes = list(sorted(map(len, components), reverse=True))
        component_sizes.append(0)
        last_component_size = metagraph.order() + 1
        num_copies = 0
        for c in component_sizes:
            if c == last_component_size:
                num_copies += 1
            else:
                if num_copies > 0:
                    print(f"{last_component_size} x {num_copies}")
                last_component_size = c
                num_copies = 1
        print("\nExample component of size 2:\n")
        for component in components:
            if len(component) == 2:
                for p_1 in component:
                    print_nicely(p_1)
                    for p_2 in metagraph.neighbors(p_1):
                        print("")
                        print_nicely(p_2)
                    break
                break
    else:
        print("Usage: 'python3 metagraph_six_six.py generate|analyze")

