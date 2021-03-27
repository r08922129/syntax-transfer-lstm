import os
import argparse
import json
import sys
from nltk import Tree
from functools import reduce
import re
import argparse

def reduceTree(output, level=None):
    '''If level is None, return full tree without terminal words'''
    out = []
    for line in sys.stdin.readlines():
        sample = preprocessCoreNLPTree(json.loads(line))

        if level:
            sample = reduceTreeLevel(sample, level)

        out.append(json.dumps(sample)+'\n')

    with open(output, 'w') as f:
        f.writelines(out)

def reduceTreeLevel(sample, level):
    
    def dfs(root, sample, out, level):

        if level and root in sample:
            out[root] = []
            for child in sample[root]:
                out[root].append(child)
                dfs(child, sample, out, level-1)

    new_sample = {}
    dfs("ROOT", sample, new_sample, level)
    return new_sample

    
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
    return out

def collectRHS(sample, cnf=False):

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

def collectRHSFromDataset(outputFile, cnf=False):

    out = set()
    for file in sys.stdin.readlines():
        file = file.strip()
        with open(file) as f:
            for line in f.readlines():
                sample = json.loads(line)
                out.update(collectSymbols(sample, cnf))

    with open(outputFile, "w") as f:
        f.write("ROOT\n")
        for i, symbol in enumerate(list(out), 1):
            f.write("{}\n".format(symbol))
    return list(out)

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
            for nonterminal in rhs:
                out.add(re.sub(r"-\d+", '', nonterminal.symbol()))

    return out

def collectSymbolsFromDataset(outputFile, cnf=False):

    out = set()
    for file in sys.stdin.readlines():
        file = file.strip()
        with open(file) as f:
            for line in f.readlines():
                sample = json.loads(line)
                out.update(collectSymbols(sample, cnf))

    with open(outputFile, "w") as f:
        f.write("[END]\n")
        f.write("ROOT\n")
        for i, symbol in enumerate(list(out), 1):
            f.write("{}\n".format(symbol))
    return list(out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_symbol", action="store_true")
    parser.add_argument("--symbol_output")

    parser.add_argument("--reduce_tree", action="store_true")
    parser.add_argument("--reduce_level", type=int)
    parser.add_argument("--reduce_output")

    args = parser.parse_args()
    # remove terminal words
    # args:
    # 
    if args.reduce_tree:
        if args.reduce_level:
            reduceTree(args.reduce_output, args.reduce_level)
        else:
            reduceTree(args.reduce_output)

    # generate set of symbols
    if args.collect_symbol:
        collectSymbolsFromDataset(args.symbol_output)