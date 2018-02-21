import networkx as nx
import graph_builder

def score_sentences_k_core(sents_A, sents_B, score_sentence_by_word):
    graph_A = graph_builder.build_graph_word(sents_A, 3)
    graph_B = graph_builder.build_graph_word(sents_B, 3)
    print("Size of graphs : " + str(len(graph_A.nodes())) + " " + str(len(graph_B.nodes())))

    dict_core_A = nx.core_number(graph_A)#k_core_decomposition.weighted_core_number(graph_A)
    k_A = max(dict_core_A.values())
    dict_core_B = nx.core_number(graph_B)#k_core_decomposition.weighted_core_number(graph_B)
    k_B = max(dict_core_B.values())

    for word in dict_core_A:
        if word in dict_core_B:
            dict_core_A[word] = dict_core_A[word] / k_A - dict_core_B[word] / k_B
        else:
            dict_core_A[word] = dict_core_A[word] / k_A

    for word in dict_core_B:
        if word in dict_core_A:
            dict_core_B[word] = dict_core_B[word] / k_B - dict_core_A[word] / k_A
        else:
            dict_core_B[word] = dict_core_B[word] / k_B

    dict_sent_A = {}
    for i, sent in enumerate(sents_A):
        dict_sent_A[i] = score_sentence_by_word(sent, dict_core_A)
    dict_sent_B = {}
    for i, sent in enumerate(sents_B):
        dict_sent_B[i] = score_sentence_by_word(sent, dict_core_B)
    return dict_sent_A, dict_sent_B
