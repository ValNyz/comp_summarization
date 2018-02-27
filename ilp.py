# -*- coding: utf-8 -*-

"""
Comparative summarization method using ILP
Comparative News Summarization Using Linear Programming (Huang et al, 2011)
__author__ : Valentin Nyzam
"""
from itertools import chain
import pulp
from pulp import lpSum
import logging
from model import comp_model
logger = logging.getLogger(__name__)


def score_sentence_ilp(*l_sents):
    ilp = comp_model.Comp_we('testWiki', l_sents)
    ilp.prepare()
    docs = []
    p_doc_name = ""
    doc_id = -1
    for sent in chain(*l_sents):
        if sent.doc != p_doc_name:
            docs.append([])
            doc_id += 1
            p_doc_name = sent.doc
            print(p_doc_name)
        docs[doc_id].append(sent.get_list_word())
    dict_idf = comp_model.make_concept_idf_dict(docs)
    # print(dict_idf)
    # dict_idf = reuters_idf_dict(l_sents, "reuters")
    for concept in ilp.w_ij[0].keys():
        ilp.w_ij[0][concept] = ilp.w_ij[0][concept]*dict_idf[concept]
    for concept in ilp.w_ij[1].keys():
        ilp.w_ij[1][concept] = ilp.w_ij[1][concept]*dict_idf[concept]
    generate_ilp_problem(l_sents, ilp.c_ij, ilp.w_ij, ilp.u_jk, ilp.s_ik,
                         ilp.l_ik)
    return None


def generate_ilp_problem(l_sents, c_ij, w_ij, u_jk, s_ik, l_ik):
    """generate_ilp_problem
    :param c_ij: list[list[concept]]: list of concept: i #doc, j #concept
    :param w_ij: list[dict{concept:weight}]: dict of concept:weight for each
    document
    :param u_jk: dict{concept_pair:weight}: dict of concept_pair:weight
    :param s_ik: list[list[list[concept]]]: list doc as a list of sentence as a
    list of concept
    """
    logger.info("Generate ILP problem")
    prob = pulp.LpProblem('summarize', pulp.LpMaximize)

    logger.info("oc_ij")
    # oc_ij
    oc_ij = pulp.LpVariable.dicts('concepts', [(i, j) for i in range(len(c_ij))
                                               for j in range(len(c_ij[i]))],
                                  0, 1, pulp.LpBinary)
    logger.info("op_jk")
    # op_jk
    op_jk = pulp.LpVariable.dicts('pairs', [(j, k) for j in range(len(c_ij[0]))
                                            for k in range(len(c_ij[1]))],
                                  0, 1, pulp.LpBinary)
    logger.info("ocs_ijk")
    # ocs_ijk
    ocs_ijk = [[[1 if j in k else 0 for k in s_ik[i]] for j in c_ij[i]] for
               i in range(len(c_ij))]
    logger.info("os_ik")
    # os_ik
    os_ik = pulp.LpVariable.dicts('sentences',
                                  [(i, k) for i in range(len(c_ij))
                                   for k in range(len(s_ik[i]))],
                                  0, 1, pulp.LpBinary)
    logger.info("Constraint")
    lambd = 0.55
    prob += lambd*pulp.lpSum([u_jk[(j, k)]*op_jk[(j, k)]
                              if (j, k) in u_jk else 0
                              for j in range(len(c_ij[0]))
                              for k in range(len(c_ij[1]))]) + (1-lambd)*lpSum(
                             [w_ij[i][c_ij[i][j]]*oc_ij[(i, j)] for i in
                              range(len(c_ij)) for j in range(len(c_ij[i]))])

    for j in range(len(c_ij[0])):
        for k in range(len(c_ij[1])):
            prob += op_jk[(j, k)] <= oc_ij[(0, j)] and op_jk[(j, k)] <= \
                    oc_ij[(1, k)]
    for j in range(len(c_ij[0])):
        for k in range(len(c_ij[1])):
            prob += pulp.lpSum([op_jk[(j, k)]]) <= 1
    for k in range(len(c_ij[1])):
        for j in range(len(c_ij[0])):
            prob += pulp.lpSum([op_jk[(j, k)]]) <= 1
    for i in range(len(c_ij)):
        for k in range(len(s_ik[i])):
            for j in range(len(c_ij[i])):
                prob += oc_ij[(i, j)] >= ocs_ijk[i][j][k]*os_ik[(i, k)]
    for i in range(len(c_ij)):
        for j in range(len(c_ij[i])):
            prob += oc_ij[(i, j)] <= pulp.lpSum([ocs_ijk[i][j][k]*os_ik[(i, k)]
                                                for k in range(len(s_ik[i]))])

    prob += pulp.lpSum([os_ik[(i, k)]*l_ik[i][k] for i in range(len(c_ij)) for
                        k in range(len(s_ik[i]))]) <= 200
    # The problem data is written to an .lp file
    # print(prob)
    prob.writeLP("comp_ilp.lp")
    logger.info("Solve ILP problem")
    prob.solve()
    # The status of the solution is printed to the screen
    print("Status:", pulp.LpStatus[prob.status])
    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        if "sentences" in v.name:
            # print(v.name, "=", v.varValue)
            if v.varValue == 1:
                vindex = v.name.split("(")[1].split(",_")
                i = int(vindex[0])
                j = int(vindex[1].split(")")[0])
                print(l_sents[i][j])
    return None
