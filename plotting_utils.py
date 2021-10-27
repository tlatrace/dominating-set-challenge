import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import random

from dominant import load_graph


def plot_graph(input_dir: Path, graph_filename: str):
    graph = load_graph(name=input_dir / graph_filename)
    plt.title(label=graph_filename)
    labels = {n: str(n) + "\n\n" + str(graph.nodes[n]["weight"]) for n in graph.nodes}
    nx.draw(graph, with_labels=True, labels=labels, font_weight="bold")


def plot_random_graph(
    input_dir: Path,
    graph_filenames_list: list,
):
    graph_filename = random.choice(graph_filenames_list)
    plt.title(graph_filename)
    graph = load_graph(input_dir / graph_filename)
    plot_graph(graph, graph_filename)


def plot_graph_with_dominating_set(
    input_dir: Path,
    graph_filename: str,
    dominating_set: list,
) -> None:
    graph = load_graph(name=input_dir / graph_filename)
    plt.title(label=graph_filename)
    non_dominating_set = [node for node in graph.nodes() if node not in dominating_set]

    # draw nodes
    pos = nx.spring_layout(graph, k=2 / (math.sqrt(len(list(graph.nodes())))), seed=42)
    nx.draw_networkx_nodes(
        graph, pos, nodelist=non_dominating_set, node_color="tab:blue"
    )
    nx.draw_networkx_nodes(graph, pos, nodelist=dominating_set, node_color="tab:red")

    # draw edges
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(
        graph, pos, edgelist=graph.edges(), width=1, alpha=0.5, edge_color="tab:green"
    )

    # draw labels
    labels = {n: str(n) + "\n\n" + str(graph.nodes[n]["weight"]) for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels)
    return
