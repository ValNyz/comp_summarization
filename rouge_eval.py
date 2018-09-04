#!/usr/bin/env python

import os
import sys
import re
import bash
import argparse
import lxml.etree as ET


def generate_html_summary(model_name, summary):
    html_summary = "<html>\n<head><title>" + model_name + \
                   "</title></head><body bgcolor=\"white\">\n"
    for i, sentence in enumerate(summary):
        html_summary += "<a name=\"" + str(i) + "\">[" + str(i) + \
                        "]</a> <a href=\"#" + str(i) + "\" id=" + \
                        str(i) + ">" + sentence + "</a>\n"
    return html_summary + "</body>\n</html>"


def write_generated_summary(generated_path, model_name, summary):
    with open(os.path.join(generated_path, model_name +
                           "_generated.html"), 'w') as f:
        f.write(generate_html_summary(model_name, summary))


def write_model_summary(model_path, model_name, summary):
    with open(os.path.join(model_path, model_name + "_model.html"), 'w') as f:
        f.write(generate_html_summary(model_name, summary))


def write_settings_xml(peer_path, output_path, model_relative_path,
                       peer_relative_path):
    root = ET.Element("ROUGE_EVAL")
    root.set("version", "1.55")

    for corpus_id, corpus_name in enumerate(os.listdir(peer_path)):
        process = ET.Element("EVAL")
        process.set("ID", "CORPUS_" + str(corpus_id))

        model_root = ET.Element("MODEL-ROOT")
        model_root.text = os.path.join(output_path, model_relative_path)
        process.append(model_root)

        peer_root = ET.Element("PEER-ROOT")
        peer_root.text = os.path.join(output_path, peer_relative_path)
        process.append(peer_root)

        input_format = ET.Element("INPUT-FORMAT")
        input_format.set("TYPE", "SEE")
        process.append(input_format)

        peers = ET.Element("PEERS")
        generated_summary = ET.Element("P")
        generated_summary.set("ID", "0")
        generated_summary.text = corpus_name + "_generated.html"
        peers.append(generated_summary)
        process.append(peers)

        models = ET.Element("MODELS")
        i = 0
        for file in os.listdir(os.path.join(output_path, model_relative_path)):
            prog = re.compile(corpus_name[:7] + ".sum")
            # prog = re.compile(corpus_name[:5] + corpus_name[-2:])
            if (prog.match(file) is not None):
                model_summary = ET.Element("M")
                model_summary.set("ID", str(i))
                model_summary.text = file
                models.append(model_summary)
                i += 1

        process.append(models)
        root.append(process)

    tree = ET.ElementTree(root)
    tree.write(os.path.join(output_path, "rouge_settings.xml"),
               pretty_print=True, xml_declaration=True, encoding="utf-8")


def execute_rouge(rouge_path, peer_path, model_path, output_path):
    if not os.path.exists(os.path.join(output_path, "Peer",)):
        os.makedirs(os.path.join(output_path, "Peer",))
    if not os.path.exists(os.path.join(output_path, "Model",)):
        os.makedirs(os.path.join(output_path, "Model",))
    for summary in os.listdir(peer_path):
        print("Loading peer : " + summary)
        summary_sent = []
        # if os.path.isfile(os.path.join(peer_path, summary)):
        with open(os.path.join(peer_path, summary)) as f:
            for sen in f:
                summary_sent.append(sen)
        write_generated_summary(os.path.join(output_path, "Peer"),
                                summary, summary_sent)
        for file in os.listdir(model_path):
            prog = re.compile(summary[:7] + ".sum")
            # prog = re.compile(summary[:5] + summary[-2:])
            if (prog.match(file) is not None):
                print("Loading models : " + file)
                model_sent = []
                with open(os.path.join(model_path, file)) as f:
                    for sen in f:
                        model_sent.append(sen)
                write_model_summary(os.path.join(output_path, "Model"),
                                    file, model_sent)
    write_settings_xml(peer_path, output_path, "Model", "Peer")

    print("Run cmd : " + os.path.join(".", rouge_path, "ROUGE-1.5.5.pl -e " +
                                      rouge_path, "data -n 3 -x -m -c 95 -r" +
                                      "1000 -f A -p 0.5 -t 0 -a " +
                                      output_path, "rouge_settings.xml"))
    sys.stdout, sys.stderr = bash.run_script(os.path.join(".", rouge_path, "ROUGE-1.5.5.pl -e " +
                                 rouge_path, "data -n 2 -x -m -c 95 -r" +
                                 "1000 -f A -p 0.5 -t 0 -a " +
                                 output_path, "rouge_settings.xml"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        help="Input model summary folder")
    parser.add_argument("-p", "--peer", type=str,
                        help="Input peer summary folder (generated)")
    parser.add_argument("-r", "--rouge", type=str, help="ROUGE path")
    args = parser.parse_args()

    model = args.model
    peer = args.peer
    rouge_path = args.rouge

    # if not os.path.exists(os.path.join(peer, "..",)):
        # os.makedirs(os.path.join(peer, ".."))
    execute_rouge(rouge_path, peer, model, os.path.join(peer, ".."))
