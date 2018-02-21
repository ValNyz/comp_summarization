# from itertools import chain
from scipy.spatial.distance import cosine
import graph.graph_builder as graph_builder
import browse_corpus
from model.bag_of_word_vector import generate_isf_dict
from model.bag_of_word_vector import generate_tf_sent_dict
from model.bag_of_word_vector import generate_dict_vect
import logging
logger = logging.getLogger(__name__)


def score_sentences_MDS(sents_A, sents_B):
    """score_sentences_MDS

    :param sents_A:
    :param sents_B:
    """
    dict_word, dict_id = browse_corpus.build_dicts(sents_A, sents_B)
    logger.info("Size of vocabulary : " + str(len(dict_word)))
    dict_tf = generate_tf_sent_dict(dict_word, dict_id, sents_A, sents_B)
    dict_isf = generate_isf_dict(dict_word, sents_A, sents_B)

    logger.debug("Len ISF dict : " + str(len(dict_isf)))
    dict_vec = generate_dict_vect(dict_word, dict_id, dict_tf, dict_isf,
                                  sents_A, sents_B)
    logger.debug("Len VEC dict : " + str(len(dict_vec)))
    G_A = graph_builder.build_graph_sents(sents_A, dict_vec,
                                          cosine_similarity, 0.1)
    ds_A = minimum_dominating_set(G_A, [])
    logger.info(ds_A)
    G_B = graph_builder.build_graph_sents(sents_B, dict_vec,
                                          cosine_similarity, 0.1)
    graph = graph_builder.update_graph_sents(G_A, G_B, dict_vec,
                                             cosine_similarity, 0.1)
    ds = minimum_dominating_set(graph, ds_A)
    logger.info(ds)
    # score sentence in ds


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
