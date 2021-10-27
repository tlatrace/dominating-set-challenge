import sys
import os
import time
import networkx as nx
import random

from time_utils import timeit
from graph_utils import (
    get_nodes_weight_dict,
    get_nodes_neighbors_dict,
    get_dominated_nodes,
    get_non_dominated_nodes,
    compute_graph_dominating_set_score,
    get_frequency_score,
    get_max_frequency_score_node,
)
from reduction_rules import (
    weighted_degree_0_rule,
    weighted_degree_1_rule_1,
    weighted_degree_1_rule_2,
    weighted_degree_2_rule,
)


@timeit
def dominant(graph: nx.Graph) -> {int}:
    """
    A Faire:
    - Ecrire une fonction qui retourne le dominant du graphe non dirigé g passé en parametre.
    - cette fonction doit retourner la liste des noeuds d'un petit dominant de g

    :param graph: le graphe est donné dans le format networkx : https://networkx.github.io/documentation/stable/reference/classes/graph.html

    """
    start_time = time.time()
    time_limit = 3.5
    iterations_limit = 20

    nodes_weight_dict = get_nodes_weight_dict(graph=graph)
    nodes_neighbors_dict = get_nodes_neighbors_dict(graph=graph)

    # 1. reduction
    nodes_to_remove, dominating_nodes, reduced_graph = reduce_graph(
        graph=graph,
        nodes_weight_dict=nodes_weight_dict,
        nodes_neighbors_dict=nodes_neighbors_dict,
    )

    # 2. construction and 3. shrinking
    dominating_set = compute_solution(
        graph=graph,
        nodes_weight_dict=nodes_weight_dict,
        reduced_graph=reduced_graph,
        dominating_nodes=dominating_nodes,
        nodes_neighbors_dict=nodes_neighbors_dict,
    )
    score = compute_graph_dominating_set_score(
        dominating_set=dominating_set, nodes_weight_dict=nodes_weight_dict
    )

    iteration_counter = 1

    while (
        time.time() - start_time < time_limit and iteration_counter < iterations_limit
    ):  # limit to 1s the computation of each graph dominating set

        # try again 2. construction and 3. shrinking in order to obtain a better score
        dominating_set_candidate = compute_solution(
            graph=graph,
            nodes_weight_dict=nodes_weight_dict,
            reduced_graph=reduced_graph,
            dominating_nodes=dominating_nodes,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
        candidate_score = compute_graph_dominating_set_score(
            dominating_set=dominating_set_candidate, nodes_weight_dict=nodes_weight_dict
        )

        if candidate_score < score:
            dominating_set = dominating_set_candidate
            score = candidate_score

        iteration_counter += 1

    print(f"{iteration_counter} iterations have been made for this graph.")

    return dominating_set


def reduce_graph(
    graph: nx.Graph,
    nodes_weight_dict: {int: {str: int}},
    nodes_neighbors_dict: {int: {int}},
) -> ({int}, {int}):
    graph_copy = graph.copy()
    nodes_neighbors_dict_copy = nodes_neighbors_dict.copy()
    nodes_to_remove_set = set()
    dominating_nodes_set = set()
    rule_0_bool, rule_11_bool, rule_12_bool, rule_2_bool = True, True, True, True
    while rule_0_bool:
        dominating_nodes = weighted_degree_0_rule(
            graph=graph_copy, nodes_neighbors_dict=nodes_neighbors_dict_copy
        )
        if dominating_nodes != set():
            for node in dominating_nodes:
                dominating_nodes_set.add(node)
                graph_copy.remove_node(node)
            nodes_neighbors_dict_copy = get_nodes_neighbors_dict(graph=graph_copy)
        else:
            rule_0_bool = False
    while rule_11_bool:
        dominating_nodes, nodes_to_remove = weighted_degree_1_rule_1(
            graph=graph_copy,
            nodes_weight_dict=nodes_weight_dict,
            nodes_neighbors_dict=nodes_neighbors_dict_copy,
        )
        if dominating_nodes != set():
            for node in dominating_nodes:
                dominating_nodes_set.add(node)
            for node_to_remove in nodes_to_remove:
                nodes_to_remove_set.add(node_to_remove)
                graph_copy.remove_node(node_to_remove)
            nodes_neighbors_dict_copy = get_nodes_neighbors_dict(graph=graph_copy)
        else:
            rule_11_bool = False
    while rule_12_bool:
        dominating_nodes, nodes_to_remove = weighted_degree_1_rule_2(
            graph=graph_copy,
            nodes_weight_dict=nodes_weight_dict,
            nodes_neighbors_dict=nodes_neighbors_dict_copy,
        )
        if dominating_nodes != set():
            for node in dominating_nodes:
                dominating_nodes_set.add(node)
            for node_to_remove in nodes_to_remove:
                nodes_to_remove_set.add(node_to_remove)
                graph_copy.remove_node(node_to_remove)
            nodes_neighbors_dict_copy = get_nodes_neighbors_dict(graph=graph_copy)
        else:
            rule_12_bool = False
    while rule_2_bool:
        dominating_nodes, nodes_to_remove = weighted_degree_2_rule(
            graph=graph_copy,
            nodes_weight_dict=nodes_weight_dict,
            nodes_neighbors_dict=nodes_neighbors_dict_copy,
        )
        if dominating_nodes != set():
            for node in dominating_nodes:
                dominating_nodes_set.add(node)
            for node_to_remove in nodes_to_remove:
                nodes_to_remove_set.add(node_to_remove)
                graph_copy.remove_node(node_to_remove)
            nodes_neighbors_dict_copy = get_nodes_neighbors_dict(graph=graph_copy)
        else:
            rule_2_bool = False

    reduced_graph = graph.copy()
    for node_to_remove in nodes_to_remove_set:
        reduced_graph.remove_node(node_to_remove)

    return nodes_to_remove_set, dominating_nodes_set, reduced_graph


def compute_solution(
    graph: nx.Graph,
    nodes_weight_dict: {int: {str: int}},
    reduced_graph: nx.Graph,
    dominating_nodes: {int},
    nodes_neighbors_dict: {int: {int}},
):
    # 2. construction
    dominating_set = construct_dominating_set(
        reduced_graph=reduced_graph,
        dominating_nodes=dominating_nodes,
        nodes_weight_dict=nodes_weight_dict,
    )

    # 3. shrinking
    dominated_nodes = get_dominated_nodes(
        graph=graph,
        dominating_nodes=dominating_set,
        nodes_neighbors_dict=nodes_neighbors_dict,
    )
    non_dominated_nodes = get_non_dominated_nodes(
        graph=graph,
        dominating_nodes=dominating_set,
        nodes_neighbors_dict=nodes_neighbors_dict,
    )
    dominating_set = shrink_dominating_set(
        graph=graph,
        dominating_set=dominating_set,
        nodes_weight_dict=nodes_weight_dict,
        nodes_neighbors_dict=nodes_neighbors_dict,
        dominated_nodes=dominated_nodes,
        non_dominated_nodes=non_dominated_nodes,
    )

    assert nx.is_dominating_set(
        graph, dominating_set
    ), "The computed set is not a dominating set."

    return dominating_set


def construct_dominating_set(
    reduced_graph: nx.Graph,
    dominating_nodes: {int},
    nodes_weight_dict: {int: {str: int}},
) -> {int}:
    dominating_nodes_copy = dominating_nodes.copy()

    # initializing non dominated nodes set
    nodes_neighbors_dict = get_nodes_neighbors_dict(reduced_graph)
    non_dominated_nodes = get_non_dominated_nodes(
        graph=reduced_graph,
        dominating_nodes=dominating_nodes_copy,
        nodes_neighbors_dict=nodes_neighbors_dict,
    )

    # add dominating nodes iteratively
    while non_dominated_nodes:
        # pick non dominated node randomly
        random_non_dominated_node = random.choice(list(non_dominated_nodes))

        # adding node in the neighborhood of random_non_dominated_node with the maximum frequency score
        dominated_nodes = get_dominated_nodes(
            graph=reduced_graph,
            dominating_nodes=dominating_nodes_copy,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
        max_frequency_score_node = get_max_frequency_score_node(
            graph=reduced_graph,
            dominating_nodes=dominating_nodes_copy,
            random_non_dominated_node=random_non_dominated_node,
            nodes_weight_dict=nodes_weight_dict,
            dominated_nodes=dominated_nodes,
            non_dominated_nodes=non_dominated_nodes,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
        dominating_nodes_copy.add(max_frequency_score_node)

        # update non_dominated_nodes
        non_dominated_nodes = update_dominated_nodes(
            non_dominated_nodes=non_dominated_nodes,
            max_frequency_score_node=max_frequency_score_node,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
    return dominating_nodes_copy


def shrink_dominating_set(
    graph: nx.Graph,
    dominating_set: {int},
    nodes_weight_dict: {int: {str: int}},
    nodes_neighbors_dict: {int: {int}},
    dominated_nodes: {int},
    non_dominated_nodes: {int},
) -> {int}:
    shrinked_dominating_set = dominating_set.copy()

    dominating_nodes_with_score_0 = {
        node
        for node in dominating_set
        if get_frequency_score(
            graph=graph,
            dominating_nodes=dominating_set,
            node=node,
            dominated_nodes=dominated_nodes,
            non_dominated_nodes=non_dominated_nodes,
            nodes_weight_dict=nodes_weight_dict,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
        == 0
    }
    while dominating_nodes_with_score_0:
        node_to_remove = random.choice(list(dominating_nodes_with_score_0))

        shrinked_dominating_set.remove(node_to_remove)
        dominating_nodes_with_score_0.remove(node_to_remove)

        # update score function of the neighbors and their neighbors
        dominating_nodes_with_score_0 = {
            node
            for node in shrinked_dominating_set
            if get_frequency_score(
                graph=graph,
                dominating_nodes=shrinked_dominating_set,
                node=node,
                dominated_nodes=dominated_nodes,
                non_dominated_nodes=non_dominated_nodes,
                nodes_weight_dict=nodes_weight_dict,
                nodes_neighbors_dict=nodes_neighbors_dict,
            )
            == 0
        }

    return shrinked_dominating_set


def update_dominated_nodes(
    non_dominated_nodes: {int},
    max_frequency_score_node: int,
    nodes_neighbors_dict: {int: {int}},
) -> {int}:
    non_dominated_nodes = non_dominated_nodes.copy()
    if max_frequency_score_node in non_dominated_nodes:
        non_dominated_nodes.remove(max_frequency_score_node)
    max_frequency_score_node_neighbors = nodes_neighbors_dict[max_frequency_score_node]
    for max_frequency_score_node_neighbor in max_frequency_score_node_neighbors:
        if max_frequency_score_node_neighbor in non_dominated_nodes:
            non_dominated_nodes.remove(max_frequency_score_node_neighbor)
    return non_dominated_nodes


# not used anymore
def purify_nodes_set(
    graph: nx.Graph,
    dominating_set: [int],
) -> [int]:
    check_dominating_set = True
    while check_dominating_set:
        removable_nodes = set()
        for dominating_node in dominating_set:
            reduced_dominating_set = set(
                [node for node in dominating_set if node != dominating_node]
            )
            if nx.is_dominating_set(graph, reduced_dominating_set):
                removable_nodes.add(dominating_node)
        if removable_nodes:
            # node_to_remove = get_highest_weight_node(graph, removable_nodes)
            node_to_remove = random.choice(list(removable_nodes))
            dominating_set = [node for node in dominating_set if node != node_to_remove]
        else:
            check_dominating_set = False
    return dominating_set


#########################################
#### Ne pas modifier le code suivant ####
#########################################


def load_graph(name):
    with open(name, "r") as f:
        state = 0
        G = None
        for l in f:
            if state == 0:  # Header nb of nodes
                state = 1
            elif state == 1:  # Nb of nodes
                nodes = int(l)

                state = 2
            elif state == 2:  # Header position
                i = 0
                state = 3
            elif state == 3:  # Position
                i += 1
                if i >= nodes:
                    state = 4
            elif state == 4:  # Header node weight
                i = 0
                state = 5
                G = nx.Graph()
            elif state == 5:  # Node weight
                G.add_node(i, weight=int(l))
                i += 1
                if i >= nodes:
                    state = 6
            elif state == 6:  # Header edge
                i = 0
                state = 7
            elif state == 7:
                if i > nodes:
                    pass
                else:
                    edges = l.strip().split(" ")
                    for j, w in enumerate(edges):
                        w = int(w)
                        if w == 1 and (not i == j):
                            G.add_edge(i, j)
                    i += 1

        return G


#########################################
#### Ne pas modifier le code suivant ####
#########################################
if __name__ == "__main__":
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])

    # un repertoire des graphes en entree doit être passé en parametre 1
    if not os.path.isdir(input_dir):
        print(input_dir, "doesn't exist")
        exit()

    # un repertoire pour enregistrer les dominants doit être passé en parametre 2
    if not os.path.isdir(output_dir):
        print(input_dir, "doesn't exist")
        exit()

    # fichier des reponses depose dans le output_dir et annote par date/heure
    output_filename = "answers_{}.txt".format(
        time.strftime("%d%b%Y_%H%M%S", time.localtime())
    )
    output_file = open(os.path.join(output_dir, output_filename), "w")

    for graph_filename in sorted(os.listdir(input_dir)):
        # importer le graphe
        g = load_graph(os.path.join(input_dir, graph_filename))

        # calcul du dominant
        D = sorted(dominant(g), key=lambda x: int(x))

        # ajout au rapport
        output_file.write(graph_filename)
        for node in D:
            output_file.write(" {}".format(node))
        output_file.write("\n")

    output_file.close()
