
"""
Source : https://searchcode.com/codesearch/view/63865049/

from python package networkx

Find the k-cores of a graph.

The k-core is found by recursively pruning nodes with degrees less than k.

See the following reference for details:

An O(m) Algorithm for Cores Decomposition of Networks
Vladimir Batagelj and Matjaz Zaversnik, 2003.
http://arxiv.org/abs/cs.DS/0310049

Adapted by : Valentin Nyzam
Base on the work of :

__author__ = "\n".join(['Dan Schult (dschult@colgate.edu)',
                        'Jason Grout (jason-sage@creativetrax.com)',
                        'Aric Hagberg (hagberg@lanl.gov)'])
"""

import networkx as nx
import graph_builder as gb

def weighted_core_number(G):
    """Return the core number for each vertex.

    A k-core is a maximal subgraph that contains nodes of weight k or more.

    The core number of a node is the largest value k of a k-core containing
    that node.

    Parameters
    ----------
    G : NetworkX graph
       A graph or directed graph

    Returns
    -------
    core_number : dictionary
       A dictionary keyed by node to the core number.

    Raises
    ------
    NetworkXError
        The k-core is not defined for graphs with self loops or parallel edges.

    Notes
    -----
    Not implemented for graphs with parallel edges or self loops.

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik, 2003.
       http://arxiv.org/abs/cs.DS/0310049
    """
    if G.is_multigraph():
        raise nx.NetworkXError(
            'MultiGraph and MultiDiGraph types not supported.')

    if G.number_of_selfloops() > 0:
        raise nx.NetworkXError(
            'Input graph has self loops; the core number is not defined.',
            'Consider using G.remove_edges_from(G.selfloop_edges()).')

    if G.is_directed():
        import itertools
        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors_iter(v), G.successors_iter(v)])
    else:
        neighbors = G.neighbors_iter

    weights = gb.graph_weights(G)
    # sort nodes by weights
    nodes = sorted(weights, key=weights.get)
    bin_boundaries = [0]
    curr_weight = 0
    for i, v in enumerate(nodes):
        if weights[v] > curr_weight:
            bin_boundaries.extend([i] * (weights[v] - curr_weight))
            curr_weight = weights[v]
    node_pos = dict((v, pos) for pos, v in enumerate(nodes))
    # initial guesses for core is degree
    core = weights
    nbrs = dict((v, set(neighbors(v))) for v in G)
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core

def weighted_k_core(G,k=None,core_number=None):
    """Return the k-core of G.

    A k-core is a maximal subgraph that contains nodes of degree k or more.

    Parameters
    ----------
    G : NetworkX graph
      A graph or directed graph
    k : int, optional
      The order of the core.  If not specified return the main core.
    core_number : dictionary, optional
      Precomputed core numbers for the graph G.

    Returns
    -------
    G : NetworkX graph
      The k-core subgraph

    Raises
    ------
    NetworkXError
      The k-core is not defined for graphs with self loops or parallel edges.

    Notes
    -----
    The main core is the core with the largest degree.

    Not implemented for graphs with parallel edges or self loops.

    For directed graphs the node degree is defined to be the
    in-degree + out-degree.

    See Also
    --------
    core_number

    References
    ----------
    .. [1] An O(m) Algorithm for Cores Decomposition of Networks
       Vladimir Batagelj and Matjaz Zaversnik,  2003.
       http://arxiv.org/abs/cs.DS/0310049
    """
    if core_number is None:
        core_number=weighted_core_number(G)
    if k is None:
        k=max(core_number.values()) # max core
    nodes=(n for n in core_number if core_number[n]>=k)
    return G.subgraph(nodes)