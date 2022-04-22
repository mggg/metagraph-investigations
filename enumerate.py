import typer
from graphillion import GraphSet
import graphillion.tutorial as tl
import networkx as nx
from networkx.readwrite import json_graph
import json

"""
Simple script to enum graph partitions
Use graphilion v1.5rc0: https://github.com/takemaru/graphillion/tree/v1.5rc0
"""

def main(n: int, m: int):
    for c, partition in enumerate(plans(n, m)):
        data = json_graph.adjacency_data(partition)
        with open(f"plans/{c}.json", "w") as f:
            json.dump(data, f)


def plans(n: int, m: int):
    universe = tl.grid(n-1, n-1)
    GraphSet.set_universe(universe)
    GraphSet.converters['to_graph'] = nx.from_edgelist
    GraphSet.converters['to_edges'] = nx.to_edgelist

    partitions = GraphSet.balanced_partitions(lower=int((n*n)/m), upper=int((n*n)/m))
    print(f"{n}x{n} -> {m}", partitions.len())

    for partition in partitions:
        yield partition # a nx object


if __name__ == "__main__":
    typer.run(main)

