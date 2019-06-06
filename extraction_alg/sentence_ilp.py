# -*- coding: utf-8 -*-

"""
Comparative summarization method using ILP
Comparative News Summarization Using Linear Programming (Huang et al, 2011)
__author__ : Valentin Nyzam
"""
# import idf
from itertools import chain
import pulp
from pulp import lpSum
import logging
from model import comp_model
# from idf import generate_idf
logger = logging.getLogger(__name__)


def score_sentence_ilp2(model, lambd, l_sents):
    d_sen_score = model.d_sen_score
    d_sen_sim = model.d_sen_sim
    s_i = []
    l_i = []
    for sen in model.d_id_sents:
        s_i.append(l_sents[sen[0]][sen[1]])
        l_i.append(len(l_sents[sen[0]][sen[1]]))

    return solve_ilp_problem(l_sents, lambd, s_i, d_sen_score, d_sen_sim,
                                l_i, model.d_id_sents)


def solve_ilp_problem(l_sents, lambd, s_i, w_i, u_jk, l_i, d_id_sents):
    """solve_ilp_problem
    First genereate ilp problem with generate_ilp_problem() or
    generate_bi_ilp_program() then solve it

    :param c_ij: list[list[concept]]: list of concept: i #doc, j #concept
    :param w_ij: list[dict{concept:weight}]: dict of concept:weight for each
    document
    :param u_jk: dict{concept_pair:weight}: dict of concept_pair:weight
    :param s_ik: list[list[list[concept]]]: list doc as a list of sentence as a
    list of concept
    """
    logger.info("Generate ILP problem")
    prob = generate_ilp_problem(l_sents, lambd, s_i, w_i, u_jk, l_i)

    # The problem data is written to an .lp file
    # prob.writeLP("comp_ilp.lp")

    logger.info("Solve ILP problem")
    try:
        prob.solve()
    except Exception:
        logger.info('Problem infeasible')
    # The status of the solution is printed to the screen
    print("Status:", pulp.LpStatus[prob.status])
    # Each of the variables is printed with it's resolved optimum value
    summary_A = []
    summary_B = []
    for v in prob.variables():
        if "sentences" in v.name:
            if v.varValue == 1:
                # print(v.name)
                vindex = int(v.name.split("_")[1])
                i = d_id_sents[vindex][0]
                j = d_id_sents[vindex][1]
                if i == 0:
                    summary_A.append(l_sents[i][j])
                else:
                    summary_B.append(l_sents[i][j])
    return summary_A, summary_B


def generate_ilp_problem(l_sents, lambd, s_i, w_i, u_jk, l_i):
    """generate_ilp_problem
    :param c_ij: list[list[concept]]: list of concept: i #doc, j #concept
    :param w_ij: list[dict{concept:weight}]: dict of concept:weight for each
    document
    :param u_jk: dict{concept_pair:weight}: dict of concept_pair:weight
    :param s_ik: list[list[list[concept]]]: list doc as a list of sentence as a
    list of concept
    """
    prob = pulp.LpProblem('summarize', pulp.LpMaximize)

    logger.info("os_ik")
    # os_i
    os_i = pulp.LpVariable.dicts('sentences',
                                  [i for i in range(len(s_i))],
                                  0, 1, pulp.LpBinary)

    logger.info("op_jk")
    # op_jk
    nb_sen_1 = len(l_sents[0])
    nb_sen_2 = len(l_sents[1])
    op_jk = pulp.LpVariable.dicts('pairs', [(j, k) for k in range(nb_sen_1, nb_sen_1+nb_sen_2)
                                            for j in range(nb_sen_1)],
                                  0, 1, pulp.LpBinary)

    logger.info("Constraint")

    prob += lambd*lpSum([u_jk[(j, k)]*op_jk[(j, k)]
                         if (j, k) in u_jk else 0
                         for k in range(nb_sen_1, nb_sen_1+nb_sen_2)
                         for j in range(nb_sen_1)]) + (1-lambd)*lpSum(
                         [w_i[i]*os_i[i] for i in range(len(s_i))])

    for tup in u_jk.keys():
        # print(tup)
        j = tup[0]
        k = tup[1]

        if j is not None and k is not None:
            # (4) and (5)
            prob += op_jk[(j, k)] <= os_i[j] and op_jk[(j, k)] <= \
                os_i[k]
            # (6) and (7)
            prob += lpSum([op_jk[(j, k)]]) <= 1
        else:
            logger.info(tup)

    # (10)
    prob += lpSum([os_i[k]*l_i[k] for k in
                   range(nb_sen_1)]) <= 100
    prob += lpSum([os_i[k]*l_i[k] for k in
                   range(nb_sen_1, nb_sen_1+nb_sen_2)]) <= 100

    return prob
