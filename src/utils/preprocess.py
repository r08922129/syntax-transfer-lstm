import os
import argparse
import json
import re
import sys

def preprocessCoreNLPTree(mapping):

    '''Remove the postfix of symbols of the output dict of CoreNLP.
    If a non-terminal symbol's right-hand-side is a terminal word, it won't be
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
            "ROOT" : ["NP", "VP"],
        }
    '''
    def childIsTerminal(root, mapping):

        for child in mapping[root]:
            if child in mapping:
                return False
        return True
  
    def removePostfix(node):
        return re.sub(r"-.*", '', node)

    out = {}
    for symbol in mapping:
        reduceSymbol = removePostfix(symbol)
        if not childIsTerminal(symbol, mapping):
            out[reduceSymbol] = []
            for child in mapping[symbol]:
                out[reduceSymbol].append(removePostfix(child))
    return out