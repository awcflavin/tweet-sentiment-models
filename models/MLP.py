from statistics import mean
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from utils.datasets import TwitterDataset
from utils.utils import pad_tweets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLPWithFastText(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, block_size=128, hidden_dim=128, num_classes=3):
        super().__init__()
        self.block_size = block_size
        self.forward_block = nn.Sequential (
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim*block_size, num_classes)
        )
    
    def forward(self, x, y=None):
        logits = self.forward_block(x)
        if y is not None:
            loss = F.cross_entropy(logits, y)
            return logits, loss
        return logits
    
    def train_model(self, train_dataset, val_dataset, optimizer, batch_size=32):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.train()
        for (x, y) in tqdm(train_dataloader, unit="batch", total=len(train_dataloader)):
            x = x.to(device, dtype = torch.float)
            y = y.to(device, dtype = torch.long)
            optimizer.zero_grad()
            predictions, loss = self(x, y=y)

            loss.backward()
            optimizer.step()

        est_train_loss_loader = DataLoader(Subset(train_dataset, range(len(val_dataset))), batch_size=batch_size, shuffle=True)
        train_loss, train_accuracy = self.estimate_loss_and_accuracy(est_train_loss_loader)
        val_loss, val_accuracy = self.estimate_loss_and_accuracy(val_dataloader)
        
        return train_loss, val_loss, train_accuracy, val_accuracy
    
    @torch.no_grad()
    def estimate_loss_and_accuracy(self, dataloader):
        self.eval()
        losses, accs = [], []
        for (x, y) in dataloader:
            x = x.to(device, dtype = torch.float)
            y = y.to(device, dtype = torch.long)
            logits, loss = self(x, y=y)
            losses.append(loss.item())
            accs.append(accuracy_score(y.cpu().numpy(), torch.argmax(logits, dim=-1).cpu().numpy()))
        return round(mean(losses), 4), round(mean(accs), 4)
    
    def eval_model(self, test_dataset, batch_size=32):
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        results = self.get_metrics(test_dataloader)
        return results

    @torch.no_grad()
    def get_metrics(self, dataloader):
        self.eval()
        losses, ys, preds = [], [], []
        for (x, y) in dataloader:
            x = x.to(device, dtype = torch.float)
            y = y.to(device, dtype = torch.long)
            ys += y.tolist()
            logits, loss = self(x, y=y)
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
        data_padded_encoded = [F.pad(tweet, (0, 0, self.block_size-len(tweet), 0), value=0) for tweet in data_encoded]
           
        return TwitterDataset(data_padded_encoded, labels)
    
class MLP(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, block_size=128, hidden_dim=128, num_classes=3):
        super().__init__()
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.forward_block = nn.Sequential (
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim*block_size, num_classes)
        )
    
    def forward(self, x, y=None):
        x = self.embedding(x)
        logits = self.forward_block(x)
        if y is not None:
            loss = F.cross_entropy(logits, y)
            return logits, loss
        return logits
    
    def train_model(self, train_dataset, val_dataset, optimizer, batch_size=32):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.train()
        for (x, y) in tqdm(train_dataloader, unit="batch", total=len(train_dataloader)):
            x = x.to(device, dtype = torch.long)
            y = y.to(device, dtype = torch.long)
            optimizer.zero_grad()
            predictions, loss = self(x, y=y)

            loss.backward()
            optimizer.step()

        est_train_loss_loader = DataLoader(Subset(train_dataset, range(len(val_dataset))), batch_size=batch_size, shuffle=True)
        train_loss, train_accuracy = self.estimate_loss_and_accuracy(est_train_loss_loader)
        val_loss, val_accuracy = self.estimate_loss_and_accuracy(val_dataloader)
        
        return train_loss, val_loss, train_accuracy, val_accuracy
    
    @torch.no_grad()
    def estimate_loss_and_accuracy(self, dataloader):
        self.eval()
        losses, accs = [], []
        for (x, y) in dataloader:
            x = x.to(device, dtype = torch.long)
            y = y.to(device, dtype = torch.long)
            logits, loss = self(x, y=y)
            losses.append(loss.item())
            accs.append(accuracy_score(y.cpu().numpy(), torch.argmax(logits, dim=-1).cpu().numpy()))
        return round(mean(losses), 4), round(mean(accs), 4)
    
    def eval_model(self, test_dataset, batch_size=32):
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        results = self.get_metrics(test_dataloader)
        return results
    
    @torch.no_grad()
    def get_metrics(self, dataloader):
        self.eval()
        losses, ys, preds = [], [], []
        for (x, y) in dataloader:
            x = x.to(device, dtype = torch.long)
            y = y.to(device, dtype = torch.long)
            ys += y.tolist()
            logits, loss = self(x, y=y)
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
        pad_token = tokenizer.get_pad_token()

        data_padded_encoded = pad_tweets(data_encoded, self.block_size, pad_token)
           
        return TwitterDataset(data_padded_encoded, labels)