from src.data.dataset import SyntaxMappingDataset
from src.data.syntax import Production
from src.utils.preprocess import collectSymbols
import json
import re
from tqdm import tqdm
import csv

def loadSymbols(pathToSymbols):

    with open(pathToSymbols) as f:
        out = [symbol.strip() for symbol in f.readlines()]

    return out

if __name__ == "__main__":

    
    symbols = loadSymbols("save/symbols")
    # build dataset
    trainDataset = SyntaxMappingDataset(
        pathToSource="syntax_data/QQPPos/train/train-src",
        pathToReference="syntax_data/QQPPos/train/train-ref",
        symbols=symbols,
    )
    