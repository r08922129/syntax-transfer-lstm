from src.data.dataset import SyntaxMappingDataset
from src.data.syntax import Production
from src.utils.preprocess import collectSymbols
import re
import torch.nn as nn
import torch
from math import sqrt

class SymbolEmbedding(nn.Module):
    
    def __init__(self, symbols, embedding_dim):
        super(SymbolEmbedding, self).__init__()
        self.embedding = nn.Embedding(len(symbols), embedding_dim, sparse=True).requires_grad_(False)
        self.symbolIndex = {
            symbol : torch.tensor(index) for index, symbol in enumerate(symbols)
        }
        self.root_embedding = self.embedding(self.symbolIndex["ROOT"])

    def get_embedding(self, symbol):
        return self.embedding(self.symbolIndex[symbol].to(self.embedding.weight.device))

class TreeLSTMCell(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, n_ary=4):
        super(TreeLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_ary = n_ary

        self.W_iou = nn.Linear(embedding_dim, 3 * hidden_dim)
        self.U_iou = nn.Linear(n_ary * hidden_dim, 3 * hidden_dim)
        
        self.W_f = nn.Linear(embedding_dim, hidden_dim)
        self.U_f = nn.Linear(n_ary * hidden_dim, n_ary * hidden_dim)
        
    def forward(self, input, last_hidden):
        '''
        Args:
            last_hidden = (h, c)
            shape:
                input: batch, embedding_dim
                h: batch, n_ary * hidden_dim
                c: batch, n_ary * hidden_dim
        '''
        h, c = last_hidden

        iou = self.W_iou(input) + self.U_iou(h) # (batch, 3 * hidden)
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        
        f = self.U_f(h).view(h.size(0), self.n_ary, self.hidden_dim) # (batch, n_ary * hidden)
        f = f + self.W_f(input).expand(self.n_ary, h.size(0), self.hidden_dim).transpose(0, 1)
        f = torch.sigmoid(f) # (batch, n_ary, hidden_dim)

        c = i * u + torch.sum(f * c.view(-1, self.n_ary, self.hidden_dim), 1)
        h = o * torch.tanh(c)

        return h, c    
        
class SyntaxTransferEncoder(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, n_ary=4):

        super(SyntaxTransferEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_ary = n_ary

        self.h0_bottom_up = nn.Parameter(torch.rand(1, n_ary * hidden_dim)).requires_grad_()
        self.c0_bottom_up = nn.Parameter(torch.rand(1, n_ary * hidden_dim)).requires_grad_()
        
        
        self.TreeLSTMCell = TreeLSTMCell(embedding_dim, hidden_dim)
        self.ChildHiddenTransform = nn.Linear(hidden_dim*n_ary, hidden_dim)

    def forward(self, syntax, embeddings, bottom_up = True):
        
        out_h, out_c = [], []
        if bottom_up:
            h, c = self.dfsBottomUp("ROOT", syntax, embeddings, out_h, out_c)
            out_h.append(h)
            out_c.append(c)

        
        out_h = torch.stack(out_h).transpose(0, 1)
        out_c = torch.stack(out_c).transpose(0, 1)

        return out_h, out_c

    def dfsBottomUp(self, node, syntax, embeddings, out_h, out_c):
        
        node_lemma = re.sub(r"-\d+", '', node)
        node_embedding = embeddings.get_embedding(node_lemma).view(1, -1) # (batch = 1, embedding_dim)

        if node not in syntax:

            h, c = self.TreeLSTMCell(node_embedding, (self.h0_bottom_up, self.c0_bottom_up))
            out_h.append(h)
            out_c.append(c)

            return h, c

        else:
            cuda = self.h0_bottom_up.device
            h = torch.zeros(1, self.n_ary * self.hidden_dim).to(cuda) # (batch = 1, n_ary * embedding_dim)
            c = torch.zeros(1, self.n_ary * self.hidden_dim).to(cuda)

            for i, child in enumerate(syntax[node]):
                sub_h, sub_c = self.dfsBottomUp(child, syntax, embeddings, out_h, out_c)
                h[:, i*self.hidden_dim:(i+1)*self.hidden_dim] = sub_h
                c[:, i*self.hidden_dim:(i+1)*self.hidden_dim] = sub_c
            
            h, c = self.TreeLSTMCell(node_embedding, (h, c))
            out_h.append(h)
            out_c.append(c)

            return h, c

class SyntaxTransferDecoder(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, n_ary=4, level=2):

        super(SyntaxTransferDecoder, self).__init__()
        self.level = level
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_ary = n_ary

        self.LSTMCell = nn.LSTMCell(embedding_dim, hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.n_ary_position_embeddings = nn.Embedding(n_ary, hidden_dim)

    def forward(self, syntax, encoder_hiddens, embeddings):
        '''
        Args:
            encoder_hiddens = (encoder_h, encoder_c)
            
            shape:
                encoder_h : (batch, num_nodes, hidden_dim)
                encoder_c : (batch, num_nodes, hidden_dim)

        '''
        encoder_h, encoder_c = encoder_hiddens
        out = []
        self.dfsTopDown("ROOT", syntax, encoder_h, (encoder_h[:,-1,:], encoder_c[:,-1,:]), embeddings, self.level, out)
        out = torch.stack(out).transpose(0, 1)

        return out

    def dfsTopDown(self, node, syntax, encoder_hiddens, last_hidden, embeddings, level, out):
        
        if level and node in syntax:

            node_lemma = re.sub(r"-\d+", '', node)
            node_embedding = embeddings.get_embedding(node_lemma).view(1, -1)
            h, c = self.LSTMCell(node_embedding, last_hidden)
            # h (batch, hidden)
            q = self.W_q(h).unsqueeze(2) # (batch, hidden, 1)
            K = self.W_k(encoder_hiddens) # (batch, seq, hidden)
            V = self.W_v(encoder_hiddens) # (batch, seq, hidden)
            attention = torch.softmax(torch.matmul(K, q), dim = 1)
            attention = torch.sum(attention * V, dim = 1) # (batch, hidden)
            h = torch.cat([h, attention], dim=1)
            h = self.linear(h)

            for i in range(self.n_ary):
                position = torch.tensor(i).to(self.n_ary_position_embeddings.weight.device)
                position_embedding = self.n_ary_position_embeddings(position)
                out.append(h + position_embedding)

            for i, child in enumerate(syntax[node]):
                self.dfsTopDown(child, syntax, encoder_hiddens, (h, c), embeddings, level-1, out)

    
class SyntaxTransferEncoderDecoder(nn.Module):
    
    def __init__(self, symbols, embedding_dim=256, hidden_dim=256, n_ary=4, decode_level=2):
    
        super(SyntaxTransferEncoderDecoder, self).__init__()

        self.decode_level = decode_level
        self.embeddings = SymbolEmbedding(symbols, embedding_dim)
        self.encoder = SyntaxTransferEncoder(embedding_dim, hidden_dim)
        self.decoder = SyntaxTransferDecoder(embedding_dim, hidden_dim, n_ary, level=decode_level)
        self.linear = nn.Linear(hidden_dim, len(symbols))

        # init parameters
        for weight in self.encoder.parameters():
            nn.init.uniform_(weight, -sqrt(1/hidden_dim), sqrt(1/hidden_dim))

        for weight in self.decoder.parameters():
            nn.init.uniform_(weight, -sqrt(1/hidden_dim), sqrt(1/hidden_dim))

        for param in self.encoder.parameters():
            param.requires_grad_(False)

    def forward(self, source_syntax, target_syntax):

        '''
        Return:
            hiddens:
                The decoder will traverse each node in the graph until the specific level reached.
                Each node will be expanded to n_ary nodes so the shape of hiddens will be
                the number of node expanded * n_ary

                shape:
                    batch, number of node expanded * n_ary, number of symbols
        '''
        h, c = self.encoder(source_syntax, self.embeddings)
        hiddens = self.decoder(target_syntax, (h, c), self.embeddings)
        hiddens = self.linear(hiddens) # (batch, number of nodes expanded * n_ary, number of symbols)

        return hiddens