import numpy as np
import networkx as nx
import random

def create_graph(n=20, p=0.2, seed=42):
    # Setting seeds so results are reproducible (important for grading)
    np.random.seed(seed)
    random.seed(seed)

    # Create a random directed graph
    G = nx.erdos_renyi_graph(n, p, directed=True, seed=seed)

    # Assign random opinion values between 0 and 1
    opinions = np.random.rand(n)

    # Store opinion inside graph nodes for easy access later
    for i in range(n):
        G.nodes[i]["opinion"] = float(opinions[i])

    return G, opinions
