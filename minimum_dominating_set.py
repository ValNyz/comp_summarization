# from itertools import chain
from scipy.spatial.distance import cosine
import graph.graph_builder as graph_builder
import browse_corpus
from model.bag_of_word_vector import generate_isf_dict
from model.bag_of_word_vector import generate_tf_sent_dict
from model.bag_of_word_vector import generate_dict_vect
import logging
import numpy as np
logger = logging.getLogger(__name__)


def score_sentences_MDS(sents_A, sents_B):
    """score_sentences_MDS

    :param sents_A:
    :param sents_B:
    """
    d_l_sen = {}
    s_A = []
    s_B = []
    for sen in sents_A:
        s_A.append(sen.get_list_word())
        d_l_sen[id(sen.get_list_word())] = len(sen)
    for sen in sents_B:
        s_B.append(sen.get_list_word())
        d_l_sen[id(sen.get_list_word())] = len(sen)
    dict_word, dict_id = browse_corpus.build_dicts(s_A, s_B)
    logger.info("Size of vocabulary : " + str(len(dict_word)))
    dict_tf = generate_tf_sent_dict(dict_word, dict_id, s_A, s_B)
    dict_isf = generate_isf_dict(dict_word, s_A, s_B)

    logger.debug("Len ISF dict : " + str(len(dict_isf)))
    dict_vec = generate_dict_vect(dict_word, dict_id, dict_tf, dict_isf,
                                  s_A, s_B)
    logger.debug("Len VEC dict : " + str(len(dict_vec)))
    G_A = graph_builder.build_graph_sents(s_A, dict_vec,
                                          cosine_similarity, 0.1)
    ds = minimum_dominating_set(G_A, [])
    logger.debug(ds)
    G_B = graph_builder.build_graph_sents(s_B, dict_vec,
                                          cosine_similarity, 0.1)
    graph = graph_builder.update_graph_sents(G_A, G_B, dict_vec,
                                             cosine_similarity, 0.1)
    ds_A = ds.copy()
    ds = minimum_dominating_set(graph, ds)
    logger.debug(ds)
    ds_B = list(set(ds)-set(ds_A))

    d_id_sen_A_score = {}
    for i in ds_A:
        d_id_sen_A_score[i] = np.sum(dict_vec[i])/d_l_sen[i]
    d_id_sen_B_score = {}
    for i in ds_B:
        d_id_sen_B_score[i] = np.sum(dict_vec[i])/d_l_sen[i]
    summary_A = []
    for s in mmr(d_id_sen_A_score, d_l_sen, dict_vec):
        for i, sen in enumerate(s_A):
            if s == id(sen):
                summary_A.append(sents_A[i])
    summary_B = []
    for s in mmr(d_id_sen_B_score, d_l_sen, dict_vec):
        for i, sen in enumerate(s_B):
            if s == id(sen):
                summary_B.append(sents_B[i])
    return summary_A, summary_B


def cosine_similarity(v1, v2):
    """cosine_similarity

    :param v1:
    :param v2:
    """
    return 1 - cosine(v1, v2)


def minimum_dominating_set(G, ds=[]):
    """minimum_dominating_set
    Generate the minimum_dominating_set of a graph (possibly giving a partly
    domininating set of the graph)
    :param G:
    :param ds:
    """
    def process_degree_with_ds(G, ds):
        logger.debug(G.nodes())
        logger.debug(ds)
        uncovered_nodes = [node for node in G.nodes()]
        for dom_node in ds:
            for node in G.neighbors(dom_node):
                if node in uncovered_nodes:
                    uncovered_nodes.remove(node)
        for dom_node in ds:
            if dom_node in uncovered_nodes:
                uncovered_nodes.remove(dom_node)
        logger.debug("Len uncovered_nodes : " + str(len(uncovered_nodes)))

        def deg(node):
            return len([n for n in G.neighbors(node) if n in uncovered_nodes])

        degrees = {node: deg(node) for node in uncovered_nodes
                   if deg(node) != 0}
        return degrees

    degrees = process_degree_with_ds(G, ds)
    # {node: G.degree(node) for node in G.nodes() if G.degree(node) != 0}
    while True:
        logger.debug("Len DEG dict : " + str(len(degrees)))
        if len(degrees) == 0:
            break
        else:
            best_node = sorted(degrees, key=degrees.get, reverse=True)[0]
            # logger.debug("best_node : " + str(best_node))
            # logger.debug("Degree best_node : " + str(degrees[best_node]))
            ds.append(best_node)
            degrees = process_degree_with_ds(G, ds)
            # logger.debug(degrees)
    logger.debug(ds)
    return ds


def mmr(d_id_sen_score, d_l_sen, d_id_sen_vec, lambd=0.7,
        sum_size=200):
    sentences = sorted(d_id_sen_score, key=d_id_sen_score.get, reverse=True)
    summary = set([sentences[0]])
    size = d_l_sen[sentences[0]]
    while size < sum_size:
        mmr = {}
        l_sen_to_del = []
        for id_sen in d_id_sen_score.keys():
            if id_sen not in summary and d_l_sen[id_sen] < (sum_size-size):
                mmr[id_sen] = lambd * d_id_sen_score[id_sen] - \
                        (1 - lambd) * \
                        cosine_similarity(d_id_sen_vec[id_sen],
                                          np.sum([d_id_sen_vec[s]
                                                  for s in summary], axis=0))
            else:
                l_sen_to_del.append(id_sen)

        if len(mmr) == 0:
            break
        selected = max(mmr, key=mmr.get)
        summary.add(selected)
        del d_id_sen_score[selected]
        for id_sen in l_sen_to_del:
            del d_id_sen_score[id_sen]
        size += d_l_sen[selected]
    return summary
