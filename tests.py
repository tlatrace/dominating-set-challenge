import networkx as nx
import random
from pathlib import Path

from dominant import (
    load_graph,
    weighted_degree_0_rule,
    weighted_degree_1_rule_1,
    weighted_degree_1_rule_2,
    weighted_degree_2_rule,
    reduce_graph,
    construct_dominating_set,
    dominant,
    get_nodes_neighbors_dict,
)

from graph_utils import is_node_dominated
from plotting_utils import plot_graph_with_dominating_set
from time_utils import timeit


INPUT_DIR = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\3A_Centrale_Supélec\COURS\ALGO-ALGORITHMIQUE_AVANCEE\concours_algo\public_data\public_dataset")
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


def test_weighted_degree_0_rule() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50)
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph=graph)
    assert weighted_degree_0_rule(graph, nodes_neighbors_dict) == {50}


def test_weighted_degree_1_rule_1() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    nodes_weight_dict = dict(graph.nodes(data=True))
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph=graph)
    assert weighted_degree_1_rule_1(graph, nodes_weight_dict, nodes_neighbors_dict) == (
        {9},
        {0},
    )  # use plot_graph() to check visually


def test_weighted_degree_1_rule_2() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50, weight=1)
    graph.add_node(51, weight=5)
    graph.add_edge(50, 9)
    graph.add_edge(51, 9)
    nodes_weight_dict = dict(graph.nodes(data=True))
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph)
    assert weighted_degree_1_rule_2(graph, nodes_weight_dict, nodes_neighbors_dict) == (
        {9},
        {0, 50, 51},
    ), f"Should be ({{9}}, {{0, 50, 51}}) but is {weighted_degree_1_rule_2(graph)}"


def test_weighted_degree_2_rule() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50, weight=40)
    graph.add_edge(50, 9)
    graph.add_edge(50, 0)
    nodes_weight_dict = dict(graph.nodes(data=True))
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph)
    assert weighted_degree_2_rule(graph, nodes_weight_dict, nodes_neighbors_dict) == (
        {9},
        {0, 50},
    ), f"Should be {{9}} but is {weighted_degree_2_rule(graph)}"


def test_reduce_graph() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50, weight=1)
    graph.add_node(51, weight=5)
    graph.add_node(52, weight=30)
    graph.add_node(53, weight=40)
    graph.add_node(54)
    graph.add_edge(50, 9)
    graph.add_edge(51, 9)
    graph.add_edge(26, 52)
    graph.add_edge(26, 53)
    graph.add_edge(52, 53)

    nodes_weight_dict = dict(graph.nodes(data=True))
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph=graph)
    nodes_to_remove, dominating_nodes, reduced_graph = reduce_graph(
        graph, nodes_weight_dict, nodes_neighbors_dict
    )
    assert (nodes_to_remove, dominating_nodes) == (
        {0, 50, 51, 52, 53},
        {9, 26, 54},
    ), f"Result should be ({{0, 50, 51, 52, 53}}, {{9, 26, 54}}) but is {reduce_graph(graph)}"


def test_is_node_linked_to_dominating_nodes() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    dominating_nodes = {3, 35}
    node_number = 3
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph)
    assert (
        is_node_dominated(
            node=node_number,
            dominating_nodes=dominating_nodes,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
        is True
    ), f"Node {node_number} should be linked to dominated set but is not."
    node_number = 4
    assert (
        is_node_dominated(
            node=node_number,
            dominating_nodes=dominating_nodes,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
        is True
    ), f"Node {node_number} should be linked to dominated set but is not."
    node_number = 43
    assert (
        is_node_dominated(
            node=node_number,
            dominating_nodes=dominating_nodes,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
        is False
    ), f"Node {node_number} shouldn't be linked to dominated set but is linked."
    dominating_nodes = {}
    node_number = 7
    assert (
        is_node_dominated(
            node=node_number,
            dominating_nodes=dominating_nodes,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
        is False
    ), f"Node {node_number} shouldn't be linked to dominated set but is linked."


def test_construct_dominating_set() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50, weight=1)
    graph.add_node(51, weight=5)
    graph.add_node(52, weight=30)
    graph.add_node(53, weight=40)
    graph.add_node(54)
    graph.add_edge(50, 9)
    graph.add_edge(51, 9)
    graph.add_edge(26, 52)
    graph.add_edge(26, 53)
    graph.add_edge(52, 53)

    nodes_weight_dict = dict(graph.nodes(data=True))
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph=graph)
    nodes_to_remove, dominating_nodes, reduced_graph = reduce_graph(
        graph, nodes_weight_dict, nodes_neighbors_dict
    )
    dominating_set = construct_dominating_set(
        reduced_graph, dominating_nodes, nodes_weight_dict
    )
    assert nx.is_dominating_set(
        graph, dominating_set
    ), f"Set {dominating_set} is not a dominating set"


def test_dominant() -> None:
    graph = load_graph(INPUT_DIR / GRAPH_FILENAME)
    graph.add_node(50, weight=1)
    graph.add_node(51, weight=5)
    graph.add_node(52, weight=30)
    graph.add_node(53, weight=40)
    graph.add_node(54, weight=56)
    graph.add_edge(50, 9)
    graph.add_edge(51, 9)
    graph.add_edge(26, 52)
    graph.add_edge(26, 53)
    graph.add_edge(52, 53)

    assert nx.is_dominating_set(
        graph, dominant(graph)
    ), f"Set {dominant(graph)} is not a dominating set"


def _test_plot(
    input_dir: Path,
    graph_filename: str,
) -> None:
    graph = load_graph(input_dir / graph_filename)
    dominating_set = dominant(graph)
    plot_graph_with_dominating_set(input_dir, graph_filename, dominating_set)


def _test_plot_random_graph(
    input_dir: Path,
    graph_filenames_list: list,
) -> None:
    graph_filename = random.choice(graph_filenames_list)
    graph = load_graph(input_dir / graph_filename)
    dominating_set = dominant(graph)
    plot_graph_with_dominating_set(input_dir, graph_filename, dominating_set)


@timeit
def _compute_global_score(
    input_dir: Path,
    graph_filenames_list: list,
) -> int:
    total_score = 0
    for graph_filename in graph_filenames_list:
        total_score += _compute_graph_dominant_set_score(input_dir, graph_filename)
    print(f"Total score for the {len(graph_filenames_list)} graphs : {total_score}.")
    return total_score


def _compute_graph_dominant_set_score(
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


# -----
# DEBUG

# test_dominant(INPUT_DIR, GRAPH_FILENAME)
# _compute_graph_dominant_set_score(INPUT_DIR, GRAPH_FILENAME)
# _compute_global_score(INPUT_DIR, GRAPH_FILENAMES_LIST)
# _test_plot(INPUT_DIR, GRAPH_FILENAME)
# _test_plot_random_graph(INPUT_DIR, GRAPH_FILENAMES_LIST)
