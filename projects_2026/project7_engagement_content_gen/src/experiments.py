from .llm import generate_post, get_sentiment_score
from .propagation import propagate
import pandas as pd

def test_multiple_messages(G, opinions, prompts):
    records = []

    for i, prompt in enumerate(prompts, start=1):
        post = generate_post(prompt)
        sentiment_value = get_sentiment_score(post)

        activated_count, updated_opinions, activated_nodes = propagate(
            G, opinions, sentiment_value, start_node=0
        )

        records.append({
            "Post Number": i,
            "Prompt": prompt,
            "Post Text": post,
            "Sentiment Score": round(sentiment_value, 3),
            "Engagement Score": activated_count,
            "Activated Nodes": activated_nodes,
            "Updated Opinions": updated_opinions
        })

    return pd.DataFrame(records)

def test_single_prompt(G, opinions, prompt, num_posts=5):
    records = []

    for i in range(1, num_posts + 1):
        post = generate_post(prompt)
        sentiment_value = get_sentiment_score(post)

        activated_count, updated_opinions, activated_nodes = propagate(
            G, opinions, sentiment_value, start_node=0
        )

        records.append({
            "Post Number": i,
            "Prompt": prompt,
            "Post Text": post,
            "Sentiment Score": round(sentiment_value, 3),
            "Engagement Score": activated_count,
            "Activated Nodes": activated_nodes,
            "Updated Opinions": updated_opinions
        })

    return pd.DataFrame(records)

def propagate(G, opinions, message, start_node=0, threshold=0.3, alpha=0.5):
    updated_opinions = opinions.copy()
    visited = set([start_node])
    queue = [start_node]
    activated_nodes = set([start_node])

    while queue:
        node = queue.pop(0)

        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                diff = abs(updated_opinions[neighbor] - message)

                if diff < threshold:
                    updated_opinions[neighbor] += alpha * (message - updated_opinions[neighbor])
                    activated_nodes.add(neighbor)
                    queue.append(neighbor)

                visited.add(neighbor)

    return len(activated_nodes), updated_opinions, activated_nodes