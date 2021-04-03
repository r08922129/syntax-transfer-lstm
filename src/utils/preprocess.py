import os
import argparse
import json
import sys
from nltk import Tree
from functools import reduce
import re
import argparse

def computeLevel(tree):
    
    def dfs(node, tree, level):
        
        if node not in tree:
            return level
        else:
            max_level = level
            for child in tree[node]:
                max_level = max(max_level, dfs(child, tree, level+1))
            return max_level

    return dfs('ROOT', tree, 0)

def computeBandWidth(tree):
    
    def dfs(node, tree):
        
        if node in tree:
            max_band = len(tree[node])
            for child in tree[node]:
                max_band = max(max_band, dfs(child, tree))
            return max_band
        else:
            return 0
        
    return dfs('ROOT', tree)

def reduceSampleBandWidth(source_file, reference_file, max_bandwidth):
    '''
    Return:
        out: List of tuple of source and reference tree
    '''
    out = []
    for source, reference in zip(source_file.readlines(), reference_file.readlines()):
        source, reference = json.loads(source), json.loads(reference)
        
        bandwidth = max(computeBandWidth(source), computeBandWidth(reference))
        if bandwidth <= max_bandwidth:
            out.append((source, reference))
    return out
    
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

def collectSymbolsFromDataset(cnf=False):

    out = set()
    for file in sys.stdin.readlines():
        file = file.strip()
        with open(file) as f:
            for line in f.readlines():
                sample = json.loads(line)
                out.update(collectSymbols(sample, cnf))

    print("[END]")
    print("ROOT")
    for symbol in list(out):
        print(symbol)

    return list(out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--remove_terminal", action="store_true")
    parser.add_argument("--collect_symbol", action="store_true")

    # reduce bandwidth
    parser.add_argument("--reduce_bandwidth", action="store_true")
    parser.add_argument("--source_file", type=str)
    parser.add_argument("--reference_file", type=str)
    parser.add_argument("--source_output", type=str)
    parser.add_argument("--reference_output", type=str)
    parser.add_argument("--max_bandwidth", type=int)

    args = parser.parse_args()

    if args.reduce_bandwidth:
        with open(args.source_file) as source_file, open(args.reference_file) as reference_file:
            out = reduceSampleBandWidth(source_file, reference_file, args.max_bandwidth)
        with open(args.source_output, 'w') as source_output, open(args.reference_output, 'w') as reference_output:
            for source, reference in out:
                source_output.write(json.dumps(source)+'\n')
                reference_output.write(json.dumps(reference)+'\n')

    # remove terminal words
    elif args.remove_terminal:
        for line in sys.stdin.readlines():
            sample = json.loads(line)
            print(json.dumps(preprocessCoreNLPTree(sample)))

    # generate set of symbols
    elif args.collect_symbol:
        collectSymbolsFromDataset()