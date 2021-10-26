import sys
import os
import time
import networkx as nx
import random


def timeit(method):
    """Decorator to time the execution of a function."""

    def timed(*args, **kw):
        start_time = time.time()
        print(f"\nStarting execution of {method.__name__}.")
        result = method(*args, **kw)
        end_time = time.time()
        n_seconds = round(end_time - start_time, 3)
        if n_seconds < 60:
            print(f"\n{method.__name__} : {n_seconds}s to execute")
        elif 60 < n_seconds < 3600:
            print(
                f"\n{method.__name__} : {n_seconds // 60}min {n_seconds % 60}s to execute"
            )
        else:
            print(
                f"\n{method.__name__} : {n_seconds // 3600}h {n_seconds % 3600 // 60}min {n_seconds // 3600 % 60}s to execute"
            )
        return result

    return timed


# @timeit
# def dominant(graph) -> list:
#     """
#     A Faire:
#     - Ecrire une fonction qui retourne le dominant du graphe non dirigé g passé en parametre.
#     - cette fonction doit retourner la liste des noeuds d'un petit dominant de g
#
#     :param graph: le graphe est donné dans le format networkx : https://networkx.github.io/documentation/stable/reference/classes/graph.html
#
#     """
#     # get exhaustive information about nodes
#     full_node_dict = get_full_node_dict(graph)
#
#     # sort nodes according to a certain criteria
#
#     # sorted_nodes_list = get_nodes_sorted_by_ratio(nodes_dict)
#     # sorted_nodes_list = get_nodes_sorted_by_neighbors_weight_ratio(nodes_dict)
#     sorted_nodes_list = get_nodes_randomly_sorted_by_ratios(full_node_dict)
#     # sorted_nodes_list = get_nodes_sorted_by_neighbors_number(nodes_dict)
#     # sorted_nodes_list = get_nodes_sorted_by_weight(nodes_dict)
#     # sorted_nodes_list = get_nodes_sorted_by_log_ratios(nodes_dict)
#
#     # creating dominating set
#     dominating_set = list()
#     counter = 0
#     while (not nx.is_dominating_set(graph, dominating_set)) and (
#         counter < len(list(graph.nodes))
#     ):
#         next_node = sorted_nodes_list[counter]
#         #  reformat to minimize the cost function : don't compute before the sorted_nodes_list
#         #  but compute it on the fly (like this we can compute with dominating_set length also)
#         if not is_node_dominated(next_node, graph, dominating_set):
#             dominating_set.append(next_node)
#
#         counter += 1
#
#     #  swap nodes if necessary
#     #  implement a dict with number of dominating vertices that dominate a given non-dominant vertex
#
#     # swap nodes from dominating set if necessary
#     # dominating_nodes_number = get_dominating_nodes_number_dict(full_node_dict, dominating_set)
#     # dominating_set, dominating_nodes_number = swap_nodes_if_necesseary(dominating_set, full_node_dict, dominating_nodes_number)
#
#     # purify dominating set by removing non necessary nodes
#     # dominating_set = purify_nodes_set(graph, dominating_set)
#
#     assert nx.is_dominating_set(
#         graph, dominating_set
#     ), "ohoh, there's an issue... : This is not a dominating set"
#
#     return sorted(dominating_set)


@timeit
def dominant(graph: nx.Graph) -> {int}:
    """
    A Faire:
    - Ecrire une fonction qui retourne le dominant du graphe non dirigé g passé en parametre.
    - cette fonction doit retourner la liste des noeuds d'un petit dominant de g

    :param graph: le graphe est donné dans le format networkx : https://networkx.github.io/documentation/stable/reference/classes/graph.html

    """

    # L = list()
    # with mp.Pool(processes=8) as pool:
    #     L.append(pool.map(f, [graph]*4))
    #
    # dominating_sets = [i for i in L[0]]
    #
    # nodes_weight_dict = get_nodes_weight_dict(graph=graph)
    # score = 1000000
    # d = dominating_sets[0]
    # for dominating_set in dominating_sets:
    #     candidate_score = compute_graph_dominating_set_score(dominating_set=dominating_set, nodes_weight_dict=nodes_weight_dict)
    #     if score > candidate_score:
    #         score = candidate_score
    #         d = dominating_set
    #
    # dominating_set = set(d)

    start_time = time.time()
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
        time.time() - start_time < 1
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


def f(graph):
    start_time = time.time()
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
    )
    score = compute_graph_dominating_set_score(
        dominating_set=dominating_set, nodes_weight_dict=nodes_weight_dict
    )

    while (
        time.time() - start_time < 1
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

    return dominating_set


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


def compute_graph_dominating_set_score(
    dominating_set: {int}, nodes_weight_dict: {int: {str: int}}
) -> int:
    score = sum(
        [
            nodes_weight_dict[dominating_node]["weight"]
            for dominating_node in dominating_set
        ]
    )
    return score


# @timeit
# def get_full_node_dict(graph: nx.Graph) -> Dict[int, Dict]:
#     return {
#         node_number: {
#             "weight": weight_dict["weight"],
#             "neighbors": get_node_neighbors(graph=graph, node_number=node_number),
#             "ratio": len(get_node_neighbors(graph=graph, node_number=node_number))
#             / weight_dict["weight"],
#             "neighbors_weight_ratio": get_node_total_neighbors_weight(
#                 graph, node_number
#             )
#             / weight_dict["weight"],
#         }
#         for node_number, weight_dict in dict(graph.nodes(data=True)).items()
#     }


# def get_nodes_sorted_by_ratio(nodes_dict: Dict[int, Dict]) -> list:
#     nodes_ratio_dict = {
#         node_number: node_dict["ratio"] for node_number, node_dict in nodes_dict.items()
#     }
#     sorted_nodes_ratio_list = sorted(
#         nodes_ratio_dict.items(), key=operator.itemgetter(1), reverse=True
#     )
#     sorted_nodes_list = [node_tuple[0] for node_tuple in sorted_nodes_ratio_list]
#     return sorted_nodes_list


# def get_nodes_sorted_by_neighbors_number(nodes_dict: Dict[int, Dict]) -> list:
#     nodes_neighbors_number_dict = {
#         node_number: len(node_dict["neighbors"])
#         for node_number, node_dict in nodes_dict.items()
#     }
#     sorted_nodes_neighbors_number_list = sorted(
#         nodes_neighbors_number_dict.items(), key=operator.itemgetter(1), reverse=True
#     )
#     sorted_nodes_list = [
#         node_tuple[0] for node_tuple in sorted_nodes_neighbors_number_list
#     ]
#     return sorted_nodes_list


# def get_nodes_sorted_by_weight(nodes_dict: Dict[int, Dict]) -> list:
#     nodes_weight_dict = {
#         node_number: node_dict["weight"]
#         for node_number, node_dict in nodes_dict.items()
#     }
#     sorted_nodes_weight_list = sorted(
#         nodes_weight_dict.items(), key=operator.itemgetter(1), reverse=True
#     )
#     sorted_nodes_list = [node_tuple[0] for node_tuple in sorted_nodes_weight_list]
#     return sorted_nodes_list


# def get_nodes_sorted_by_neighbors_weight_ratio(nodes_dict: Dict[int, Dict]) -> list:
#     nodes_neighbors_weight_ratio_dict = {
#         node_number: node_dict["neighbors_weight_ratio"]
#         for node_number, node_dict in nodes_dict.items()
#     }
#     sorted_nodes_neighbors_weight_ratio_list = sorted(
#         nodes_neighbors_weight_ratio_dict.items(),
#         key=operator.itemgetter(1),
#         reverse=True,
#     )
#     sorted_nodes_list = [
#         node_tuple[0] for node_tuple in sorted_nodes_neighbors_weight_ratio_list
#     ]
#     return sorted_nodes_list


# def get_nodes_randomly_sorted_by_ratios(nodes_dict: Dict[int, Dict]) -> list:
#     sorted_by_ratio_nodes_list = get_nodes_sorted_by_ratio(nodes_dict)
#     sorted_by_neighbors_weight_ratio_nodes_list = (
#         get_nodes_sorted_by_neighbors_weight_ratio(nodes_dict)
#     )
#     return get_randomly_lists_fusion(
#         sorted_by_ratio_nodes_list, sorted_by_neighbors_weight_ratio_nodes_list
#     )


# def get_nodes_sorted_by_log_ratios(nodes_dict: Dict[int, Dict]) -> list:
#     nodes_ratio_dict = {
#         node_number: math.log(node_dict["weight"]) / len(node_dict["neighbors"])
#         for node_number, node_dict in nodes_dict.items()
#     }
#     sorted_nodes_ratio_list = sorted(
#         nodes_ratio_dict.items(), key=operator.itemgetter(1), reverse=True
#     )
#     sorted_nodes_list = [node_tuple[0] for node_tuple in sorted_nodes_ratio_list]
#     return sorted_nodes_list


# def get_randomly_lists_fusion(l1: list, l2: list) -> list:
#     assert len(l1) == len(l2)
#     assert sorted(l1) == sorted(l2)  # check if same elements in the two lists
#
#     # with probability 50%, pick next node in nodes sorted by weight list or nodes sorted by weight ratio list
#     sorted_nodes_list = list()
#     counter1 = 0
#     counter2 = 0
#     while counter1 < len(l1) and counter2 < len(l2):
#         random_number = random.random()
#         if random_number > 0.5:
#             next_node = l1[counter1]
#             counter1 += 1
#             if next_node not in sorted_nodes_list:
#                 sorted_nodes_list.append(next_node)
#         else:
#             next_node = l2[counter2]
#             counter2 += 1
#             if next_node not in sorted_nodes_list:
#                 sorted_nodes_list.append(next_node)
#     if counter1 == len(l1):
#         for node in l2[counter2:]:
#             if node not in sorted_nodes_list:
#                 sorted_nodes_list.append(node)
#     else:
#         for node in l1[counter1:]:
#             if node not in sorted_nodes_list:
#                 sorted_nodes_list.append(node)
#
#     assert len(sorted_nodes_list) == len(l1), f"{len(sorted_nodes_list)} != {len(l1)}"
#     assert sorted(sorted_nodes_list) == sorted(l1)
#
#     return sorted_nodes_list


# def get_node_neighbors(
#     graph: nx.Graph,
#     node_number: int,
# ) -> list:
#     return [edge[1] for edge in list(graph.edges(node_number))]


# todo : delete
def get_node_neighbors(
    graph: nx.Graph,
    node: int,
) -> set:
    return set([edge[1] for edge in list(graph.edges(node))])


def get_nodes_neighbors_dict(
    graph: nx.Graph,
) -> {int: set}:
    return {
        node: set([edge[1] for edge in list(graph.edges(node))]) for node in graph.nodes
    }


# todo : delete
def get_node_degree(
    graph: nx.Graph,
    node_number: int,
) -> int:
    node_neighbors = get_node_neighbors(graph, node_number)
    return len(node_neighbors)


# TODO : reformat this to get rid of the "weight"
def get_nodes_weight_dict(
    graph: nx.Graph,
) -> {int: {str: int}}:
    return dict(graph.nodes(data=True))


def weighted_degree_0_rule(
    graph: nx.Graph, nodes_neighbors_dict: {int: {int}}
) -> {int}:
    dominating_nodes = set()
    for node in graph.nodes:
        if len(nodes_neighbors_dict[node]) == 0:  # node of degree 0
            dominating_nodes.add(node)
    return dominating_nodes


def weighted_degree_1_rule_1(
    graph: nx.Graph,
    nodes_weight_dict: {int: {str: int}},
    nodes_neighbors_dict: {int: {int}},
) -> ({int}, {int}):
    dominating_nodes = set()
    nodes_to_remove = set()
    for node in graph.nodes:
        node_neighbors = nodes_neighbors_dict[node]
        node_neighbors_of_degree_1 = [
            node_neighbor
            for node_neighbor in node_neighbors
            if len(nodes_neighbors_dict[node_neighbor]) == 1  # node of degree 1
        ]
        if len(node_neighbors_of_degree_1) == 1:  # check if has a neighbor
            node_neighbor = node_neighbors_of_degree_1[0]
            if (
                nodes_weight_dict[node_neighbor]["weight"]
                > nodes_weight_dict[node]["weight"]
            ):
                dominating_nodes.add(node)
                for node_to_remove in node_neighbors_of_degree_1:
                    nodes_to_remove.add(node_to_remove)
    return dominating_nodes, nodes_to_remove


def get_neighbors_weight_sum(
    nodes_list: [int], nodes_weight_dict: {int: {str: int}}
) -> int:
    neighbors_weight_sum = 0
    for node_neighbor in nodes_list:
        neighbors_weight_sum += nodes_weight_dict[node_neighbor]["weight"]
    return neighbors_weight_sum


def weighted_degree_1_rule_2(
    graph: nx.Graph,
    nodes_weight_dict: {int: {str: int}},
    nodes_neighbors_dict: {int: {int}},
) -> ({int}, {int}):
    dominating_nodes = set()
    nodes_to_remove = set()
    for node in graph.nodes:
        node_neighbors = nodes_neighbors_dict[node]
        node_neighbors_of_degree_1 = [
            node_neighbor
            for node_neighbor in node_neighbors
            if len(nodes_neighbors_dict[node_neighbor]) == 1  # node of degree 1
        ]
        if node_neighbors_of_degree_1:  # check if some neighbor have degree 1
            neighbors_weight_sum = get_neighbors_weight_sum(
                nodes_list=node_neighbors_of_degree_1,
                nodes_weight_dict=nodes_weight_dict,
            )
            if neighbors_weight_sum > nodes_weight_dict[node]["weight"]:
                dominating_nodes.add(node)
                for node_to_remove in node_neighbors_of_degree_1:
                    nodes_to_remove.add(node_to_remove)
    return dominating_nodes, nodes_to_remove


def weighted_degree_2_rule(
    graph: nx.Graph,
    nodes_weight_dict: {int: {str: int}},
    nodes_neighbors_dict: {int: {int}},
) -> ({int}, {int}):
    dominating_nodes = set()
    nodes_to_remove = set()
    for node in graph.nodes:
        if len(nodes_neighbors_dict[node]) == 2:  # node of degree 2
            node_neighbors = nodes_neighbors_dict[node]
            for node_neighbor in node_neighbors:
                if len(nodes_neighbors_dict[node_neighbor]) == 2:
                    second_neighbor = next(iter(node_neighbors - {node_neighbor}))
                    if second_neighbor in nodes_neighbors_dict[node_neighbor]:
                        if (
                            nodes_weight_dict[node_neighbor]["weight"]
                            > nodes_weight_dict[second_neighbor]["weight"]
                            and nodes_weight_dict[node]["weight"]
                            > nodes_weight_dict[second_neighbor]["weight"]
                        ):
                            dominating_nodes.add(second_neighbor)
                            nodes_to_remove.add(node_neighbor)
                            nodes_to_remove.add(node)
    return dominating_nodes, nodes_to_remove


def get_node_total_neighbors_weight(
    graph: nx.Graph,
    node_number: int,
) -> int:
    node_neighbors = get_node_neighbors(graph, node_number)
    nodes_weight_dict = dict(graph.nodes(data=True))
    total_weight = 0
    for node in node_neighbors:
        total_weight += nodes_weight_dict[node]["weight"]
    return total_weight


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


# TODO : define this
def get_node_frequency(graph: nx.Graph, node: int) -> int:
    return 1


def get_frequency_score(
    graph: nx.Graph,
    dominating_nodes: {int},
    node: int,
    dominated_nodes: {int},
    non_dominated_nodes: {int},
    nodes_weight_dict: {int: {str: int}},
    nodes_neighbors_dict: {int: {int}},
) -> int:
    frequency_score = 0
    if node in dominating_nodes:
        # c1_nodes : dominated nodes that would become non dominated by removing node_number from dominating nodes

        dominating_nodes_reduced = dominating_nodes.copy()
        dominating_nodes_reduced.remove(node)

        c1_nodes = set()
        for dominated_node in dominated_nodes:
            if not is_node_dominated(
                node=dominated_node,
                dominating_nodes=dominating_nodes_reduced,
                nodes_neighbors_dict=nodes_neighbors_dict,
            ):
                c1_nodes.add(dominated_node)

        for c1_node in c1_nodes:
            frequency_score += (
                get_node_frequency(graph=graph, node=c1_node)
                / nodes_weight_dict[node]["weight"]
            )
            # frequency_score += 1 / nodes_weight_dict[node_number]["weight"]
    else:  # case where node_number is not a dominating node
        # c2_nodes : non dominated nodes that would become dominated by adding node_number into dominating nodes

        dominating_nodes_increased = dominating_nodes.copy()
        dominating_nodes_increased.add(node)

        c2_nodes = set()
        for non_dominated_node in non_dominated_nodes:
            if is_node_dominated(
                node=non_dominated_node,
                dominating_nodes=dominating_nodes_increased,
                nodes_neighbors_dict=nodes_neighbors_dict,
            ):
                c2_nodes.add(non_dominated_node)

        for c2_node in c2_nodes:
            frequency_score += (
                get_node_frequency(graph=graph, node=c2_node)
                / nodes_weight_dict[node]["weight"]
            )
            # frequency_score += 1 / nodes_weight_dict[node_number]["weight"]

    return frequency_score


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


def get_dominated_nodes(
    graph: nx.Graph, dominating_nodes: {int}, nodes_neighbors_dict: {int: {int}}
) -> {int}:
    dominated_nodes = set()
    for node in graph.nodes:
        if is_node_dominated(
            node=node,
            dominating_nodes=dominating_nodes,
            nodes_neighbors_dict=nodes_neighbors_dict,
        ):
            dominated_nodes.add(node)
    return dominated_nodes


def get_non_dominated_nodes(
    graph: nx.Graph,
    dominating_nodes: {int},
    nodes_neighbors_dict: {int: {int}},
) -> {int}:
    non_dominated_nodes = set()
    for node in graph.nodes:
        if not is_node_dominated(
            node=node,
            dominating_nodes=dominating_nodes,
            nodes_neighbors_dict=nodes_neighbors_dict,
        ):
            non_dominated_nodes.add(node)
    return non_dominated_nodes


def get_max_frequency_score_node(
    graph: nx.Graph,
    dominating_nodes: {int},
    random_non_dominated_node: int,
    nodes_weight_dict: {int: {str: int}},
    dominated_nodes: {int},
    non_dominated_nodes: {int},
    nodes_neighbors_dict: {int: {int}},
) -> int:
    """
    Return the node in the neighborhood of random_non_dominated_node with the maximum frequency score
    """

    nodes_frequency_score_dict = get_nodes_frequency_score_dict(
        graph=graph,
        dominating_nodes=dominating_nodes,
        node=random_non_dominated_node,
        nodes_weight_dict=nodes_weight_dict,
        dominated_nodes=dominated_nodes,
        non_dominated_nodes=non_dominated_nodes,
        nodes_neighbors_dict=nodes_neighbors_dict,
    )
    max_frequency_score_node = max(
        nodes_frequency_score_dict.keys(), key=lambda k: nodes_frequency_score_dict[k]
    )
    return max_frequency_score_node


def get_nodes_frequency_score_dict(
    graph: nx.Graph,
    dominating_nodes: {int},
    node: int,
    nodes_weight_dict: {int: {str: int}},
    dominated_nodes: {int},
    non_dominated_nodes: {int},
    nodes_neighbors_dict: {int: {int}},
):
    nodes_frequency_score_dict = {
        node: get_frequency_score(
            graph=graph,
            dominating_nodes=dominating_nodes,
            node=node,
            dominated_nodes=dominated_nodes,
            non_dominated_nodes=non_dominated_nodes,
            nodes_weight_dict=nodes_weight_dict,
            nodes_neighbors_dict=nodes_neighbors_dict,
        )
        for node in nodes_neighbors_dict[node].union({node})
    }
    return nodes_frequency_score_dict


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


def is_node_dominated(
    node: int, dominating_nodes: {int}, nodes_neighbors_dict: {int: {int}}
) -> bool:
    is_linked = False
    if node in dominating_nodes:
        is_linked = True
    else:
        node_neighbors = nodes_neighbors_dict[node]
        for node_neighbor in node_neighbors:
            if node_neighbor in dominating_nodes:
                is_linked = True
    return is_linked


# todo : test this
def get_highest_weight_node(graph: nx.Graph, nodes_list: {int}) -> int:
    node_weight_dict = {
        node: weight
        for node, weight_dict in dict(graph.nodes(data=True)).items()
        for key, weight in weight_dict.items()
    }
    highest_weight_node = max(
        node_weight_dict.keys(), key=lambda k: node_weight_dict[k]
    )
    return highest_weight_node


# todo : test this
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
            print(f"{len(removable_nodes)} nodes removed.")
        else:
            check_dominating_set = False
    return dominating_set


# @timeit
# def get_dominating_nodes_number_dict(
#     full_node_dict: dict, dominating_set: [int]
# ) -> {int: int}:
#     dominating_nodes_number = dict()
#     for node in full_node_dict.keys():
#         dominating_nodes_number[node] = 0
#         for neighbor_node in full_node_dict[node]["neighbors"]:
#             if neighbor_node in dominating_set:
#                 dominating_nodes_number[node] += 1
#         if node in dominating_set:
#             dominating_nodes_number[node] += 1
#     return dominating_nodes_number


# to do : make a 2-2 swap
# def swap_nodes(
#     old_dominating_node: int,
#     new_dominating_node: int,
#     dominating_set: [int],
#     full_node_dict: dict,
#     dominating_nodes_number: {int: int},
# ) -> ([int], {int: int}):
#     dominating_set.remove(old_dominating_node)
#     dominating_set.append(new_dominating_node)
#     for old_dominating_node_neighbor in full_node_dict[old_dominating_node][
#         "neighbors"
#     ]:
#         if (
#             old_dominating_node_neighbor
#             not in full_node_dict[new_dominating_node]["neighbors"]
#         ):
#             dominating_nodes_number[old_dominating_node_neighbor] -= 1
#     for new_dominating_node_neighbor in full_node_dict[new_dominating_node][
#         "neighbors"
#     ]:
#         if (
#             new_dominating_node_neighbor
#             not in full_node_dict[old_dominating_node]["neighbors"]
#         ):
#             dominating_nodes_number[new_dominating_node_neighbor] += 1
#     return dominating_set, dominating_nodes_number


# def swap_nodes_if_necesseary(
#     dominating_set: [int],
#     full_node_dict: dict,
#     dominating_nodes_number: {int: int},
# ):
#     for dominating_node in dominating_set:
#         if len(full_node_dict[dominating_node]["neighbors"]) == 2:
#             neighbors_have_two_dominants = True
#             for dominating_node_neighbor in full_node_dict[dominating_node][
#                 "neighbors"
#             ]:
#                 if dominating_nodes_number[dominating_node_neighbor] == 1:
#                     neighbors_have_two_dominants = False
#             if neighbors_have_two_dominants:
#                 for dominating_node_neighbor in full_node_dict[dominating_node][
#                     "neighbors"
#                 ]:
#                     if (
#                         full_node_dict[dominating_node]["weight"]
#                         > full_node_dict[dominating_node_neighbor]["weight"]
#                     ):
#                         dominating_set, dominating_nodes_number = swap_nodes(
#                             dominating_node,
#                             dominating_node_neighbor,
#                             dominating_set,
#                             full_node_dict,
#                             dominating_nodes_number,
#                         )
#                     break
#     return dominating_set, dominating_nodes_number


# @timeit
# def swap_dominating_set_nodes_if_necessary(
#     full_node_dict: dict,
#     dominating_set: [int],
# ) -> [int]:
#     dominating_nodes_number_dict = get_dominating_nodes_number_dict(
#         full_node_dict, dominating_set
#     )
#     for dominating_node in dominating_set:
#         neighbors_local_weight = 0
#         for dominating_node_neighbor in full_node_dict[dominating_node]["neighbors"]:
#             if (
#                 dominating_nodes_number_dict[dominating_node_neighbor] == 1
#             ):  # then it would need to become a dominating node
#                 neighbors_local_weight += full_node_dict[dominating_node_neighbor][
#                     "weight"
#                 ]
#         if neighbors_local_weight < full_node_dict[dominating_node]["weight"]:
#             dominating_set.remove(dominating_node)
#             for node_neighbor in full_node_dict[dominating_node]["neighbors"]:
#                 if dominating_nodes_number_dict[node_neighbor] == 1:
#                     dominating_set.append(node_neighbor)
#             dominating_nodes_number_dict = get_dominating_nodes_number_dict(
#                 full_node_dict, dominating_set
#             )
#     return dominating_set


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
