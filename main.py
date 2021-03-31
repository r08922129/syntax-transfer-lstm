from src.data.dataset import SyntaxMappingDataset
from src.data.syntax import Production
from src.utils.preprocess import collectSymbols
from src.model.model import *
from src.utils.weights import get_weight_by_counts
import json
import re
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import csv
import argparse
import pickle
import os

def loadSymbols(pathToSymbols):

    with open(pathToSymbols) as f:
        out = [symbol.strip() for symbol in f.readlines()]
    return out

def computeLossByLevel(output, target, indexesEachLevel, criteriaOfEachLevel):

    loss = 0
    for level, indexes in enumerate(indexesEachLevel):
        loss += criteriaOfEachLevel[level](output[indexes], target[indexes])
    return loss

def evaluate(model, dataloader, n_ary, decode_level, criteriaOfEachLevel, printResult=False):

    def outputAccuracyEachLevel(accuracyEachLevel):
        out = []
        for level, acc in enumerate(accuracyEachLevel):
            out.append("Level {}: {}, ".format(level, acc))
        return ' '.join(out)

    model.eval()
    loss = 0
    correctEachLevel = [0] * decode_level
    countEachLevel = [0] * decode_level

    with torch.no_grad():
        for batch in tqdm(dataloader):
            for sample in batch:
                # source to reference
                output = model(sample["source"]["syntax"], sample["reference"]["syntax"])
                output = output.view(-1, output.size(2))
                target = sample["reference"]["label"].to(cuda)
                loss += computeLossByLevel(output, target, sample["reference"]["level"], criteriaOfEachLevel)
                accumulateAccuracyEachLayer(correctEachLevel=correctEachLevel,
                                            countEachLevel=countEachLevel,
                                            predictIndexes=output.argmax(1),
                                            target=target, 
                                            indexesEachLevel=sample["reference"]["level"], 
                                            n_ary=n_ary)
                if printResult:
                    print("Target:", indexesToSymbols(target, total_symbols, sample["reference"]["level"]))
                    print("Predict:", indexesToSymbols(output.argmax(1), total_symbols, sample["reference"]["level"]))
                    print()
                # reference to source
                output = model(sample["reference"]["syntax"], sample["source"]["syntax"])
                output = output.view(-1, output.size(2))
                target = sample["source"]["label"].to(cuda)
                loss += computeLossByLevel(output, target, sample["source"]["level"], criteriaOfEachLevel)
                accumulateAccuracyEachLayer(correctEachLevel=correctEachLevel,
                                            countEachLevel=countEachLevel,
                                            predictIndexes=output.argmax(1),
                                            target=target,
                                            indexesEachLevel=sample["source"]["level"],
                                            n_ary=n_ary)
                if printResult:
                    print("Target:", indexesToSymbols(target, total_symbols, sample["source"]["level"]))
                    print("Predict:", indexesToSymbols(output.argmax(1), total_symbols, sample["source"]["level"]))
                    print()

    loss = loss / (2 * len(dataloader)* dataloader.batch_size * decode_level)

    accuracyEachLevel = []
    for correct, count in zip(correctEachLevel, countEachLevel):
        accuracyEachLevel.append(correct/count)

    print("Evaluation Results. loss: {:.5f} accuracy at each level: {}".format(loss, outputAccuracyEachLevel(accuracyEachLevel)))
    model.train()
    acc = torch.tensor(correctEachLevel).sum()/torch.tensor(countEachLevel).sum()
    return acc

def accumulateAccuracyEachLayer(correctEachLevel, countEachLevel, predictIndexes, target, indexesEachLevel, n_ary):

    for level, indexes in enumerate(indexesEachLevel):
        predictLevelIndexes = predictIndexes[indexes].view(-1, n_ary)
        targetLevelIndexes = target[indexes].view(-1, n_ary)
        correctEachLevel[level] += (predictLevelIndexes == targetLevelIndexes).all(1).sum()
        countEachLevel[level] += predictLevelIndexes.size(0)

def indexesToSymbols(predictIndexes, symbols, indexesEachLevel):
    '''
    Args:
        predictIndexes:
            shape:
                (node number)

        symbols: a List of symbols.
        indexesEachLevel: a List of indexes at each level.
                
    '''
    out = []
    for indexes in indexesEachLevel:
        level_predict = []
        for index in predictIndexes[indexes]:
            level_predict.append(symbols[index])
        out.append(level_predict)
    return out

def train(model, train_dataloader, val_dataloader, optimizer,
          criteriaOfEachLevel, n_ary, decode_level, eval_point, save_path):

    total_batch = len(train_dataloader)
    eval_point = int(eval_point * total_batch)
    total_acc = 0
    for i, batch in enumerate(train_dataloader):

        batch_size = len(batch)
        optimizer.zero_grad()
        loss = 0
        acc = 0

        for j, sample in enumerate(batch):
            # source to reference
            output = model(sample["source"]["syntax"], sample["reference"]["syntax"])
            output = output.view(-1, output.size(2))
            target = sample["reference"]["label"].to(cuda)
            loss += computeLossByLevel(output, target, sample["reference"]["level"], criteriaOfEachLevel)
            acc += ((output.argmax(1) == target).view(-1, n_ary).all(1)).sum()

            # reference to source
            output = model(sample["reference"]["syntax"], sample["source"]["syntax"])
            output = output.view(-1, output.size(2))
            target = sample["source"]["label"].to(cuda)
            loss += computeLossByLevel(output, target, sample["source"]["level"], criteriaOfEachLevel)
            acc += ((output.argmax(1) == target).view(-1, n_ary).all(1)).sum()

        total_acc += acc
        acc = acc / (2 * batch_size * decode_level)
        loss = loss / (2 * batch_size * decode_level)
        print("Processing {:05d}/{} batch. loss: {:.5f} accuracy: {:.4f}\r".format(i, total_batch, loss, acc), end='')

        loss.backward()
        optimizer.step()

        if i % eval_point == 0:
            print("\nStart evaluation...\n")
            eval_acc = evaluate(model, val_dataloader, n_ary, decode_level, criteriaOfEachLevel)
            path = os.path.join(save_path, "model_{}_{}_{:.3f}.pt".format(iteration, i/eval_point, eval_acc))
            torch.save(model.state_dict(), path)


    print("Training Accuracy: {}".format(total_acc/len(train_dataloader.dataset)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("symbols")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--end_symbol_weight", type=float, default=0.25)
    parser.add_argument("--decode_level", type=int, default=2)
    parser.add_argument("--n_ary", type=int, default=4)
    parser.add_argument("--eval_point", type=float, default=0.25)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--embedding_size", type=int,default=256)
    parser.add_argument("--hidden_size", type=float, default=256)

    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()

    cuda = torch.device('cuda')

    total_symbols = loadSymbols(args.symbols)
    # build dataset
    trainDataset = SyntaxMappingDataset(
        pathToSource=os.path.join(args.dataset, "train/src"),
        pathToReference=os.path.join(args.dataset, "train/ref"),
        symbols=total_symbols,
        decode_level=args.decode_level,
        n_ary=args.n_ary,
    )
    valDataset = SyntaxMappingDataset(
        pathToSource=os.path.join(args.dataset, "val/src"),
        pathToReference=os.path.join(args.dataset, "val/src"),
        symbols=total_symbols,
        decode_level=args.decode_level,
        n_ary=args.n_ary,
    )
    testDataset = SyntaxMappingDataset(
        pathToSource=os.path.join(args.dataset, "val/src"),
        pathToReference=os.path.join(args.dataset, "val/src"),
        symbols=total_symbols,
        decode_level=args.decode_level,
        n_ary=args.n_ary,
    )
    train_dataloader = DataLoader(trainDataset, batch_size=args.batch_size, collate_fn=trainDataset.collate_fn)
    val_dataloader = DataLoader(valDataset, batch_size=args.batch_size, collate_fn=valDataset.collate_fn)
    test_dataloader = DataLoader(testDataset, batch_size=args.batch_size, collate_fn=testDataset.collate_fn)

    weights = get_weight_by_counts(trainDataset.counts)
    criteriaOfEachLevel = [CrossEntropyLoss(weight=weight.to(cuda)) for weight in weights]

    model = SyntaxTransferEncoderDecoder(
        total_symbols,
        embedding_dim = args.embedding_size,
        hidden_dim = args.hidden_size,
        n_ary = args.n_ary,
        decode_level = args.decode_level,
        ).to(cuda)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    if args.save_path:
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)

    if args.load_path:
        model.load_state_dict(torch.load(args.load_path))

    if args.train:
        print("Training Start:")
        for iteration in range(args.iterations):
            print("{}-th iteration.".format(iteration))
            train(model, train_dataloader, val_dataloader, optimizer, criteriaOfEachLevel,
                  args.n_ary, args.decode_level, args.eval_point, args.save_path)
    
    if args.evaluate:
        evaluate(model, val_dataloader, args.n_ary, args.decode_level, criteriaOfEachLevel, printResult=True)