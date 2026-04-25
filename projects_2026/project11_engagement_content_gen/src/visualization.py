import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(G, opinions, activated_nodes=None, title="Graph"):
    # Basic visualization to understand spread visually
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    # Highlight activated nodes with larger size
    node_sizes = [
        700 if activated_nodes and node in activated_nodes else 350
        for node in G.nodes()
    ]

    nx.draw(
        G,
        pos,
        node_color=opinions,
        cmap=plt.cm.viridis,
        node_size=node_sizes,
        with_labels=True,
        ax=ax
    )

    plt.title(title)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(opinions), vmax=max(opinions)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    plt.show()
