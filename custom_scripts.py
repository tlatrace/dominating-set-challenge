import networkx as nx
from pathlib import Path

from submission.dominant import load_graph, dominant, get_highest_weight_node, dominant, timeit
    # get_randomly_lists_fusion, get_dominating_nodes_number_dict, get_full_node_dict, swap_dominating_set_nodes_if_necessary,
import random
import matplotlib.pyplot as plt
import math

INPUT_DIR = Path(
    r"C:\Users\thiba\OneDrive - CentraleSupelec\3A_Centrale_Supélec\COURS\ALGO-ALGORITHMIQUE_AVANCEE\concours_algo\public_data\public_dataset"
)
GRAPH_FILENAME = "graph_50_50"
GRAPH_FILENAMES_LIST = [
    "graph_100_100",
    "graph_100_1000",
    "graph_100_250",
    "graph_100_500",
    "graph_250_1000",
    "graph_250_250",
    "graph_250_500",
    "graph_500_1000",
    "graph_500_500",
    "graph_50_1000",
    "graph_50_250",
    "graph_50_50",
    "graph_50_500",
]


def plot_graph(
    graph: nx.Graph,
    graph_filename: str
):
    plt.title(graph_filename)
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
    graph = load_graph(input_dir / graph_filename)
    plt.title(graph_filename)
    non_dominating_set = [node for node in graph.nodes() if node not in dominating_set]

    # draw nodes
    pos = nx.spring_layout(graph, k=2/(math.sqrt(len(list(graph.nodes())))), seed=42)
    nx.draw_networkx_nodes(graph, pos, nodelist=non_dominating_set, node_color="tab:blue")
    nx.draw_networkx_nodes(graph, pos, nodelist=dominating_set, node_color="tab:red")

    # draw edges
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), width=1, alpha=0.5, edge_color="tab:green")

    # draw labels
    labels = {n: str(n) + "\n\n" + str(graph.nodes[n]["weight"]) for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels)
    return


def test_plot(
        input_dir: Path,
        graph_filename: str,
) -> None:
    graph = load_graph(input_dir / graph_filename)
    dominating_set = dominant(graph)
    plot_graph_with_dominating_set(input_dir, graph_filename, dominating_set)


def test_plot_random_graph(
        input_dir: Path,
        graph_filenames_list: str,
) -> None:
    graph_filename = random.choice(graph_filenames_list)
    graph = load_graph(input_dir / graph_filename)
    dominating_set = dominant(graph)
    plot_graph_with_dominating_set(input_dir, graph_filename, dominating_set)


def compute_graph_dominant_set_score(
        input_dir: Path,
        graph_filename: str,
) -> int:
    graph = load_graph(input_dir / graph_filename)
    print("`\nGraph : ", graph_filename)
    dominant_set = dominant(graph)
    score = sum(
        [
            weight_dict["weight"]
            for node_number, weight_dict in dict(graph.nodes(data=True)).items()
            if node_number in dominant_set
        ]
    )
    return score


@timeit
def compute_global_score(
        input_dir: Path,
        graph_filenames_list: list,
) -> int:
    total_score = 0
    for graph_filename in graph_filenames_list:
        total_score += compute_graph_dominant_set_score(input_dir, graph_filename)
    print(f"Total score for the {len(graph_filenames_list)} graphs : {total_score}.")
    return total_score


def get_node_weight(node: tuple):
    return node[1]["weight"]


def test_dominant(
    input_dir: Path,
    graph_filename: str,
):
    graph = load_graph(input_dir / graph_filename)
    return dominant(graph)


def test_get_highest_weight_node(
        input_dir: Path,
        graph_filename: str,
):
    graph = load_graph(input_dir / graph_filename)
    nodes_list = graph.nodes()
    highest_weight_node = get_highest_weight_node(graph, nodes_list)
    return highest_weight_node


# def test_get_random_list_fusion():
#     L1 = [1, 2, 3, 4, 5, 6]
#     L2 = [6, 5, 4, 3, 2, 1]
#     return get_randomly_lists_fusion(L1, L2)


def test_get_dominating_nodes_number():
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    dominating_set = dominant(graph)
    full_node_dict = get_full_node_dict(graph)
    dominating_nodes_number_dict = get_dominating_nodes_number_dict(full_node_dict, dominating_set)
    return dominating_nodes_number_dict


def test_swap_dominating_set_nodes_if_necessary():
    # turn swaping off in dominant
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    dominating_set = dominant(graph)
    print(compute_graph_dominant_set_score(graph, dominating_set))
    full_node_dict = get_full_node_dict(graph)
    dominating_set = swap_dominating_set_nodes_if_necessary(full_node_dict, dominating_set)
    print(compute_graph_dominant_set_score(graph, dominating_set))
    plot_graph_with_dominating_set(INPUT_DIR, GRAPH_FILENAME)
    return dominating_set


# -----
# DEBUG

# plot_graph(INPUT_DIR, GRAPH_FILENAME)
# plot_random_graph(INPUT_DIR, GRAPH_FILENAMES_LIST)
# get_node_neighbors(INPUT_DIR, GRAPH_FILENAME, 6)
# test_dominant(INPUT_DIR, GRAPH_FILENAME)
# compute_graph_dominant_set_score(INPUT_DIR, GRAPH_FILENAME)
# compute_graph_dominant_set_score(INPUT_DIR, "graph_500_1000")
# compute_graph_dominant_set_score(INPUT_DIR, random.choice(GRAPH_FILENAMES_LIST))
compute_global_score(INPUT_DIR, GRAPH_FILENAMES_LIST)
# test_plot(INPUT_DIR, GRAPH_FILENAME)
# test_plot_random_graph(INPUT_DIR, GRAPH_FILENAMES_LIST)
# test_get_highest_weight_node(INPUT_DIR, GRAPH_FILENAME)