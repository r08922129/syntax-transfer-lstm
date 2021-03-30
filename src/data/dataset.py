from torch.utils.data import Dataset
import re
import json
import random
import torch

class SyntaxMappingDataset(Dataset):

    def __init__(self, pathToSource, pathToReference, symbols, decode_level, n_ary):
        '''
        Args:
            pathToFile: a file with each line in json format
                {
                    "sentence" : Dict<symbol, List[child]>
                    "reference" : Dict<symbol, List[child]>
                }

            symbols: a list of all the symbols which
                provide the mapping between a symbols and the index.
                
                The index will be used to select the embedding of each symbols.

        '''
        self.symbols = symbols
        self.symbolIndex = {
            symbol : i for i, symbol in enumerate(symbols)
        }
        self.decode_level = decode_level
        self.counts = None
        if self.symbolIndex["[END]"] != 0:
            raise Exception("Symbol [END] must have index 0.")
        self.samplePairs = []
        with open(pathToSource) as src, open(pathToReference) as ref:

            data = list(zip(src.readlines(), ref.readlines()))

            for source, reference in data:

                source = json.loads(source)
                reference = json.loads(reference)

                samplePair = {}
                samplePair["source"] = {
                    "syntax" : source,
                    "label" : [],
                    "level": []
                }
                samplePair["reference"] = {
                    "syntax" : reference,
                    "label" : [],
                    "level": []
                }
                self.samplePairs.append(samplePair)

        self._generateLabel(decode_level, n_ary)

    def __len__(self):
        return len(self.samplePairs)

    def __getitem__(self, index):
        return self.samplePairs[index]

    def _generateLabel(self, decode_level, n_ary):
        '''Add label to each sample in self.syntaxPairs
        '''
        self.counts = [[0 for i in range(len(self.symbols))] for j in range(decode_level)]

        def reduceLevel(node_levels, decode_level):
            node_levels = torch.tensor(node_levels, dtype=torch.long)
            new_level = []
            for l in range(decode_level):
                new_level.append((node_levels==l).nonzero().flatten())

            return new_level

        def dfs(root, sample, level, decode_level):

            if level < decode_level and root in sample['syntax']:

                for child in sample['syntax'][root]:
                    child = re.sub(r"-\d+", '', child)
                    if child not in self.symbolIndex:
                        raise Exception("Child symbol {} is not in the setting.".format(child))

                    sample["label"].append(self.symbolIndex[child])
                    sample["level"].append(level)
                    self.counts[level][self.symbolIndex[child]] += 1

                diff = n_ary - len(sample['syntax'][root])
                for i in range(diff):
                    sample["label"].append(self.symbolIndex["[END]"])
                    sample["level"].append(level)
                    self.counts[level][self.symbolIndex['[END]']] += 1


                for child in sample["syntax"][root]:
                    dfs(child, sample, level+1, decode_level)

        for samplePair in self.samplePairs:

            dfs('ROOT', samplePair["source"], 0, decode_level)
            dfs('ROOT', samplePair["reference"], 0, decode_level)

            samplePair["source"]["label"] = torch.tensor(samplePair["source"]["label"], dtype=torch.long)
            samplePair["reference"]["label"] = torch.tensor(samplePair["reference"]["label"], dtype=torch.long)

            samplePair["source"]["level"] = reduceLevel(samplePair["source"]["level"], decode_level)
            samplePair["reference"]["level"] = reduceLevel(samplePair["reference"]["level"], decode_level)


    
    def collate_fn(self, samples):
        return samples

if __name__ == "__main__":

    dataset = SyntaxMappingDataset("syntax_data/test", ["ROOT"])
    print(dataset[0])