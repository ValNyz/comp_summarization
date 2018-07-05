# -*- coding: utf-8 -*-

"""
__author__ : Valentin Nyzam
"""
import os
from itertools import chain
import model.comp_model as comp_model
import browse_corpus
import time


def generate_idf(file_name, data_path=None, update=False):
    idf_file = file_name + ".idf"
    dict_idf = {}
    if os.path.exists(idf_file):
        with open(idf_file, 'r') as f:
            l = int(next(f))
            for line in f:
                values = line.split("\t")
                dict_idf[tuple(values[0].split())] = float(values[1].rstrip())
        if data_path != None and update:
            docs = load_multicorpus(data_path)
            dict_idf = comp_model.make_concept_idf_dict(docs,
                                                        dict_idf=dict_idf, l=l)
            os.remove(idf_file)
            with open(idf_file, 'w') as f:
                f.write(str(l + len(docs)) + '\n')
                for concept in dict_idf.keys():
                    f.write(' '.join(concept) + '\t' + str(dict_idf[concept]) +
                            '\n')
            
        return dict_idf
    else:
        docs = load_multicorpus(data_path)
        dict_idf = comp_model.make_concept_idf_dict(docs)
        with open(idf_file, 'w') as f:
            f.write(str(len(docs)) + '\n')
            for concept in dict_idf.keys():
                f.write(' '.join(concept) + '\t' + str(dict_idf[concept]) +
                        '\n')
        return dict_idf


def load_multicorpus(data_path):
    docs = []
    p_doc_name = ""
    doc_id = -1
    for c_id in os.listdir(data_path):
        if 'summary' in c_id or 'rouge_settings.xml' in c_id \
        or 'Model' in c_id or 'Peer' in c_id:
            continue
        print(data_path)
        print(c_id)
        path = os.path.join(data_path, c_id)
        s_A, sents_A = browse_corpus.load_sents(path, c_id + '-A')
        s_B, sents_B = browse_corpus.load_sents(path, c_id + '-B')
        for sent in chain(sents_A, sents_B):
            if sent.doc != p_doc_name:
                docs.append([])
                doc_id += 1
                p_doc_name = sent.doc
                print(p_doc_name)
            docs[doc_id].append(sent.get_list_word())
    return docs


def parse_options():
    """
    set up command line parser options
    """
    from optparse import OptionParser
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)

    parser.add_option('-i', '--input-path', dest='input_path', type='str',
                      help='path of input files')
    parser.add_option('-f', '--file_name', dest='file_name', type='str',
                      help='name of the file to read or write')
    parser.add_option('-u', '--update', dest='update', action="store_true",
                      help='update the model load with data')

    return parser.parse_args()


if __name__ == '__main__':
    options, task = parse_options()

    start = time.process_time()
    if options.input_path is None:
        generate_idf(options.file_name)
    elif options.update is None:
        generate_idf(options.file_name, options.input_path)
    else:
        generate_idf(options.file_name, options.input_path, options.update)
    end = time.process_time()
    print('Time elapsed : ' + str(end-start))
