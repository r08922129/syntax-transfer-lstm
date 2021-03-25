import os
import argparse
import json
import sys
from nltk import Tree
from functools import reduce
import re

def preprocessCoreNLPTree(mapping):

    '''If a non-terminal symbol's right-hand-side is a terminal word, it won't be
    a key in the result.
    
    Args:
        mapping:
            eg. {
                "ROOT" : ["NP-1", "VP-1"],
                "NP-1" : ["TERMINAL_A"],
                "VP_1" : ["TERMINAL_B"],
            }

    Return:
        eg. {
            "ROOT" : ["NP-1", "VP-1"],
        }
    '''
    def childIsTerminal(root, mapping):

        for child in mapping[root]:
            if child in mapping:
                return False
        return True

    out = {}
    for symbol in mapping:
        if not childIsTerminal(symbol, mapping):
            out[symbol] = []
            for child in mapping[symbol]:
                out[symbol].append(child)
    return json.dumps(out)

def collectSymbols(sample, cnf=False):

    def dfs(root, sample):
        if root not in sample:
            return Tree(root, [])
        else:
            return Tree(root, [dfs(child, sample) for child in sample[root]])

    tree = dfs("ROOT", sample)
    if cnf:
        Tree.chomsky_normal_form(tree)

    out = set()
    for production in tree.productions():
        lhs, rhs = production.lhs(), production.rhs()
        if rhs:
            rhs = ' '.join([re.sub(r"-\d+", '', nonterminal.symbol()) for nonterminal in rhs])
        out.add(rhs)

    return out

def collectSymbolsFromDataset(files, outputFile, cnf=False):

    out = set()
    for file in files:
        with open(file) as f:
            for line in f.readlines():
                sample = json.loads(line)
                out.update(collectSymbols(sample, cnf))
    with open(outputFile, "w") as f:
        f.write("ROOT\n")
        for i, symbol in enumerate(list(out), 1):
            f.write("{}\n".format(symbol))
    return list(out)


if __name__ == "__main__":

    # remove terminal words
    # for line in sys.stdin.readlines():
    #     print(preprocessCoreNLPTree(json.loads(line)))

    # generate set of symbols
    files = [
        "syntax_data/QQPPos/train/train-src",
        "syntax_data/QQPPos/train/train-ref",
        "syntax_data/QQPPos/val/val-src",
        "syntax_data/QQPPos/val/val-ref",
        "syntax_data/QQPPos/test/test-src",
        "syntax_data/QQPPos/test/test-ref",
    ]
    collectSymbolsFromDataset(files, "symbols")