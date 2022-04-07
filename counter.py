import typer
from graphillion import GraphSet
import graphillion.tutorial as tl

def main(n: int, m: int):
    universe = tl.grid(n-1, n-1)
    GraphSet.set_universe(universe)
    print(f"{n}x{n} -> {m}", GraphSet.balanced_partitions(lower=int((n*n)/m), upper=int((n*n)/m)).len())

if __name__ == "__main__":
    typer.run(main)
