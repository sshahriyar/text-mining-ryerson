def propagate(G, opinions, message, start_node=0, threshold=0.3, alpha=0.5):
    # Copy opinions so original data stays unchanged
    updated_opinions = opinions.copy()

    # We simulate spread using BFS-like traversal
    visited = set([start_node])
    queue = [start_node]

    # Track which nodes actually get influenced
    activated_nodes = set([start_node])

    while queue:
        node = queue.pop(0)

        # Check neighbors of current node
        for neighbor in G.neighbors(node):
            if neighbor not in visited:

                # Compare how far neighbor opinion is from message
                diff = abs(updated_opinions[neighbor] - message)

                # If difference is small, node gets influenced
                if diff < threshold:
                    # Update opinion (weighted combination)
                    updated_opinions[neighbor] = (
                        alpha * message + (1 - alpha) * updated_opinions[neighbor]
                    )
                    queue.append(neighbor)
                    activated_nodes.add(neighbor)

                visited.add(neighbor)

    # Return how many nodes got influenced
    return len(activated_nodes), updated_opinions, activated_nodes
