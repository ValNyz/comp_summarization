# -*- coding: utf-8 -*-

"""
Build a graph based on a list of str
__author__ : Valentin Nyzam
"""

import networkx as nx
# import numpy as np
import logging
logger = logging.getLogger('__name__')


def build_graph_sents(l_sents, dict_vec, similarity, threshold):
    """

    :param l_sents:
    :param similarity:
    :return:
    """
    graph = nx.Graph()
    for i in range(len(l_sents)):
        sent_i = l_sents[i]
        graph.add_node(id(sent_i))
        for j in range(i+1, len(l_sents)):
            sent_j = l_sents[j]
            sim = similarity(dict_vec[id(sent_i)], dict_vec[id(sent_j)])
            if sim > threshold:
                graph.add_edge(id(sent_i), id(sent_j), weight=sim)
        # graph.nodes[i]['vec'] = dict_vec[i]
        # logger.debug(len(graph.nodes[i]['vec']))
    return graph


def update_graph_sents(G, H, dict_vec, similarity, threshold):
    graph = G.copy(G)
    graph.add_nodes_from(H)
    for i in G.nodes():
        for j in H.nodes():
            logger.debug(i)
            # logger.debug(graph[i])
            sim = similarity(dict_vec[i], dict_vec[j])
            if sim > threshold:
                graph.add_edge(i, j, weight=sim)
    return graph


def build_graph_word(l_sents, sliding_window):
    """
    build graph of word for the input list of sentences
    :param l_sents: list: list of sentence: [list of [list of integer]]
    :param sliding_window: int: size of the sliding window
    :return: graph
    """
    # print("Size of the sliding_window: " + str(sliding_window))
    # print("Creating the graph of words for collection...")

    graph = nx.Graph()

    for sent in l_sents:
        add_sent_to_graph(graph, sent, sliding_window)

    return graph


def add_sent_to_graph(graph, sent, sliding_window):
    # size = 0
    if len(sent) == 1:
        graph.add_node(sent[0])
    for i in range(0, len(sent)):
        for j in range(i, min(i+sliding_window+1, len(sent))):
            if i != j and sent[i] != sent[j]:
                graph_add_edge(graph, sent[i], sent[j], basic_weight)
                # if size != graph.size() and graph.size() % 200 == 0 :
                #     size = graph.size()
                #     print("Current size of the graph : " + str(graph.size()))


def graph_add_node(graph, word):
    if word not in graph:
        graph.add_node(str(word))


def graph_add_edge(graph, word1, word2, weighted_function):
    if not graph.has_edge(word1, word2):
        graph.add_edge(word1, word2, weight=weighted_function(word1, word2))
    else:
        graph[word1][word2]['weight'] += weighted_function(word1, word2)


def graph_weights(G):
    """
    return a dictionary of the weight of each node
    :param G: graph: input graph
    :return: dict : dictionnary of node
    """
    dico_node_weight = {}
    for n in G:
        dico_node_weight[n] = node_weight(G, n)
    return dico_node_weight


def node_weight(G, n):
    w = 0
    for adj_node in G[n]:
        w += G[n][adj_node]['weight']
    return w


def basic_weight(*args):
    return 1


def print_values_graph(graph):
    print(graph.edges(data=True))
    print(graph.nodes(data=True))
    for n in graph:
        print(str(n) + " " + str(node_weight(graph, n)))
    print()
    for n in graph.nodes():
        print(str(n) + " " + str(graph.degree(n)))
        # + " " + str(graph.in_degree(n)) + " " + str(graph.out_degree(n)))
