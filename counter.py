import typer
from graphillion import GraphSet
import graphillion.tutorial as tl
import networkx as nx

def main(n: int, m: int):
    universe = tl.grid(n-1, n-1)
    GraphSet.set_universe(universe)
    GraphSet.converters['to_graph'] = nx.from_edgelist
    GraphSet.converters['to_edges'] = nx.to_edgelist

    partitions = GraphSet.balanced_partitions(lower=int((n*n)/m), upper=int((n*n)/m))
    print(f"{n}x{n} -> {m}", partitions.len())

if __name__ == "__main__":
    typer.run(main)
