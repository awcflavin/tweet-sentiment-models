from abc import abstractmethod
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import math
from statistics import mean
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from utils.datasets import TwitterDatasetForLSTM, pad_collate
from sklearn.metrics import classification_report, accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# base class that defines the train method
class LstmBase(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x, lens, y=None):
        raise NotImplementedError
    
    def train_model(self, train_dataset, val_dataset, optimizer, batch_size):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
        self.train()
        for (x, y, length) in tqdm(train_dataloader, unit="batch", total=len(train_dataloader)):
            x = x.to(device, dtype = torch.long)
            y = y.to(device, dtype = torch.long)
            optimizer.zero_grad()
            predictions, loss = self(x, length, y=y)

            loss.backward()
            optimizer.step()

        train_subset = Subset(train_dataset, range(min(1000, len(val_dataset))))
        est_train_loss_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
        
        train_loss, train_accuracy = self.estimate_loss_and_accuracy(est_train_loss_loader)
        val_loss, val_accuracy = self.estimate_loss_and_accuracy(val_dataloader)
        
        return train_loss, val_loss, train_accuracy, val_accuracy
    
    @torch.no_grad()
    def estimate_loss_and_accuracy(self, dataloader):
        self.eval()
        losses, accs = [], []
        for (x, y, lens) in dataloader:
            x = x.to(device, dtype = torch.long)
            y = y.to(device, dtype = torch.long)
            logits, loss = self(x, lens, y=y)
            losses.append(loss.item())
            accs.append(accuracy_score(y.cpu().numpy(), torch.argmax(logits, dim=-1).cpu().numpy()))
        return round(mean(losses), 4), round(mean(accs), 4)
    
    def eval_model(self, test_dataset, batch_size=32):
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
        results = self.get_metrics(test_dataloader)
        return results
    
    @torch.no_grad()
    def get_metrics(self, dataloader):
        self.eval()
        losses, ys, preds = [], [], []
        for (x, y, lens) in dataloader:
            x = x.to(device, dtype = torch.long)
            y = y.to(device, dtype = torch.long)
            ys += y.tolist()
            logits, loss = self(x, lens, y=y)
            preds += torch.argmax(logits, dim=-1).tolist()
            losses.append(loss.item())

        ys = np.array(ys)
        preds = np.array(preds)
        target_names = ['dissapointed', 'happy', 'angry']
        metrics = classification_report(ys, preds, target_names=target_names, digits=3)
        
        return metrics, mean(losses)
    
    def get_dataset(self, data, labels, tokenizer, add_special_tokens=False, remove_emojis=False, tag=""):
        data_encoded = tokenizer.encode_many(data, add_special_tokens=add_special_tokens,
                                             remove_emojis=remove_emojis, tag=tag)
        return TwitterDatasetForLSTM(data_encoded, labels)

class LstmFromScratchWithWC(nn.Module):
    def __init__(self, input_sz, hidden_sz, reverse=False):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.reverse = reverse
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.P = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 2))
        self.P_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, lens):
        """Assumes x is of shape (batch, sequence length, feature)"""
        # if this is a bidirectional cell, we reverse the sequence
        if self.reverse:
            x = torch.flip(x, dims=[1])

        bs, seq_sz, _ = x.size()
        hidden_seq = []
        h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                    torch.zeros(bs, self.hidden_size).to(x.device))
        
        hs = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # For efficiency, we can compute the 4 gates in one go as a single matrix multiplication
            gates = x_t @ self.W + c_t @ self.U + self.bias
            peep = c_t @ self.P
            i_t, f_t, g_t = (
                torch.sigmoid(gates[:, :hs] + torch.tanh(peep[:, :hs])), # input
                torch.sigmoid(gates[:, hs:hs*2] + torch.tanh(peep[:, hs:])), # forget
                torch.tanh(gates[:, hs*2:hs*3])
            )
            c_t = f_t * c_t + (i_t * g_t) # cell state
            peep_o = c_t @ self.P_o
            o_t = torch.sigmoid(gates[:, hs*3:] + torch.tanh(peep_o)) # output
            h_t = o_t * torch.tanh(c_t) # hidden state
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        # input is padded on the right using torch.nn.utils.rnn.pad_sequence
        # therefore when reversed, the last element is the last non-pad element
        if self.reverse:
            return hidden_seq[:, -1, :]
        
        # if not reversed we get the last non-pad element by indexing using the lengths of the inputs
        batch_indices = torch.arange(bs).to(x.device)
        lens = torch.tensor(lens).to(x.device)
        output = hidden_seq[batch_indices, lens-1, :]
        return output

# LSTM implementation from scratch with working connections (slow)
class LSTMWithWC(LstmBase):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=128, num_classes=3, bidirectional=False, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm_forward = LstmFromScratchWithWC(emb_dim, hidden_dim)
        self.lstm_backward = LstmFromScratchWithWC(emb_dim, hidden_dim, reverse=True) if bidirectional else None
        final_layer_input = 2*hidden_dim if bidirectional else hidden_dim
        self.output_block = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_layer_input, num_classes)
        )
        
    def forward(self, x, lens, y=None):
        x = self.embedding(x)
        out = self.lstm_forward(x, lens)
        if self.lstm_backward:
            backward_out = self.lstm_backward(x, lens)
            out = torch.cat((out, backward_out), dim=1)
        logits = self.output_block(out)
        if y is not None:
            loss = F.cross_entropy(logits, y)
            return logits, loss
        return logits

# LSTM model that uses PyTorch LSTM (fast)
class LSTM(LstmBase):
    def __init__(self, vocab_size, emb_dim=32, hidden_dim=128, num_classes=3, bidirectional=False, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.output_size = num_classes
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        final_layer_input = hidden_dim if not bidirectional else 2*hidden_dim
        self.output_block = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_layer_input, self.output_size)
        )
        
    def forward(self, x, length, y=None):
        x = self.embedding(x)

        x1 = torch.nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(x1)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        output_lengths = output_lengths - 1
        logits = self.output_block(output[torch.arange(output.shape[0]), output_lengths, :])

        if y is not None:
            loss = F.cross_entropy(logits, y)
            return logits, loss
        return logits