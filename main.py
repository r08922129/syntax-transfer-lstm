from src.data.dataset import SyntaxMappingDataset
from src.data.syntax import Production
from src.utils.preprocess import collectSymbols
from src.model.model import *
import json
import re
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import csv
import argparse
import pickle

def loadSymbols(pathToSymbols):

    with open(pathToSymbols) as f:
        out = [symbol.strip() for symbol in f.readlines()]
    return out

def evaluate(model, dataloader, n_ary, decode_level):

    model.eval()
    loss = 0
    acc = 0
    print("\nStart evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            for sample in batch:
                # source to reference
                output = model(sample["source"]["syntax"], sample["reference"]["syntax"])
                output = output.view(-1, output.size(2))
                target = sample["reference"]["label"]
                loss += criteria(output, target)
                acc += ((output.argmax(1) == target).view(-1, n_ary).sum(1) == n_ary).sum()
                # reference to source
                output = model(sample["reference"]["syntax"], sample["source"]["syntax"])
                output = output.view(-1, output.size(2))
                target = sample["source"]["label"]
                loss += criteria(output, target)
                acc += ((output.argmax(1) == target).view(-1, n_ary).sum(1) == n_ary).sum()

    acc = acc / (2 * dataloader.batch_size * decode_level)
    loss = loss / (2 * dataloader.batch_size * decode_level)
    print("Evaluation Results. loss: {:.5f} accuracy: {:.4f}".format(loss, acc))
    model.train()

def indexToSymbol(index, symbols):
    out = []
    for idx in index:
        out.append(symbols[idx])
    return out

def train(model, train_dataloader, val_dataloader, optimizer, criteria, n_ary, decode_level, eval_point):

    total_batch = len(train_dataloader)
    eval_point = int(eval_point * total_batch)
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
            loss += criteria(output, target)
            acc += ((output.argmax(1) == target).view(-1, n_ary).all(1)).sum()
            print("Predict", indexToSymbol(output.argmax(1), train_dataloader.dataset.symbols))
            print("Target", indexToSymbol(target, train_dataloader.dataset.symbols))

            # reference to source
            output = model(sample["reference"]["syntax"], sample["source"]["syntax"])
            output = output.view(-1, output.size(2))
            target = sample["source"]["label"].to(cuda)
            loss += criteria(output, target)
            acc += ((output.argmax(1) == target).view(-1, n_ary).all(1)).sum()
            print("Predict", indexToSymbol(output.argmax(1), train_dataloader.dataset.symbols))
            print("Target", indexToSymbol(target, train_dataloader.dataset.symbols))

        acc = acc / (2 * batch_size * decode_level)
        loss = loss / (2 * batch_size * decode_level)
        print("Processing {:05d}/{} batch. loss: {:.5f} accuracy: {:.4f}\r".format(i, total_batch, loss, acc), end='')

        loss.backward()
        # for param in model.parameters():
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        if i and i % eval_point == 0:
            evaluate(model, val_dataloader, n_ary, decode_level)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--end_symbol_weight", type=float, default=0.25)
    parser.add_argument("--decode_level", type=int, default=2)
    parser.add_argument("--n_ary", type=int, default=4)
    parser.add_argument("--eval_point", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--embedding_size", type=int,default=256)
    parser.add_argument("--hidden_size", type=float, default=128)
    args = parser.parse_args()

    cuda = torch.device('cuda')

    total_symbols = loadSymbols("save/symbols")
    # build dataset
    trainDataset = SyntaxMappingDataset(
        pathToSource="syntax_data/QQPPos_small/train/src",
        pathToReference="syntax_data/QQPPos_small/train/ref",
        symbols=total_symbols,
        decode_level=args.decode_level,
        n_ary=args.n_ary,
    )
    valDataset = SyntaxMappingDataset(
        pathToSource="syntax_data/QQPPos_small/val/src",
        pathToReference="syntax_data/QQPPos_small/val/ref",
        symbols=total_symbols,
        decode_level=args.decode_level,
        n_ary=args.n_ary,
    )
    testDataset = SyntaxMappingDataset(
        pathToSource="syntax_data/QQPPos_small/test/src",
        pathToReference="syntax_data/QQPPos_small/test/ref",
        symbols=total_symbols,
        decode_level=args.decode_level,
        n_ary=args.n_ary,
    )
    train_dataloader = DataLoader(trainDataset, batch_size=args.batch_size, collate_fn=trainDataset.collate_fn)
    val_dataloader = DataLoader(valDataset, batch_size=args.batch_size, collate_fn=valDataset.collate_fn)
    test_dataloader = DataLoader(testDataset, batch_size=args.batch_size, collate_fn=testDataset.collate_fn)

    weight = torch.ones(len(total_symbols)).to(cuda)
    weight[0] = args.end_symbol_weight
    criteria = CrossEntropyLoss(weight=weight)
    model = SyntaxTransferEncoderDecoder(total_symbols, args.embedding_size, args.hidden_size).to(cuda)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    print("Training Start:")
    for i in range(args.iterations):

        print("{}-th iteration.".format(i))
        train(model, train_dataloader, val_dataloader, optimizer, criteria, args.n_ary,
         args.decode_level, args.eval_point)
        # evaluate
        break