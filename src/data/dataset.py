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
                    "label" : []
                }
                samplePair["reference"] = {
                    "syntax" : reference,
                    "label" : []
                }
                self.samplePairs.append(samplePair)

        self.generateLabel(decode_level, n_ary)

    def __len__(self):
        return len(self.samplePairs)

    def __getitem__(self, index):
        return self.samplePairs[index]

    def generateLabel(self, decode_level, n_ary):
        '''Add label to each sample in self.syntaxPairs
        '''
        def dfs(root, sample, level):

            if level and root in sample['syntax']:

                for child in sample['syntax'][root]:
                    child = re.sub(r"-\d+", '', child)
                    if child not in self.symbolIndex:
                        raise Exception("Child symbol {} is not in the setting.".format(child))

                    sample["label"].append(self.symbolIndex[child])

                diff = n_ary - len(sample['syntax'][root])
                for i in range(diff):
                    sample["label"].append(self.symbolIndex["[END]"])

                for child in sample["syntax"][root]:
                    dfs(child, sample, level-1)

        for samplepair in self.samplePairs:

            dfs('ROOT', samplepair["source"], decode_level)
            dfs('ROOT', samplepair["reference"], decode_level)

            samplepair["source"]["label"] = torch.tensor(samplepair["source"]["label"], dtype=torch.long)
            samplepair["reference"]["label"] = torch.tensor(samplepair["reference"]["label"], dtype=torch.long)
    
    def collate_fn(self, samples):
        return samples

if __name__ == "__main__":

    dataset = SyntaxMappingDataset("syntax_data/test", ["ROOT"])
    print(dataset[0])