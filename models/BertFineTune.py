import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from utils.datasets import TwitterDatasetForBert
from transformers import DistilBertModel
from tqdm import tqdm
from statistics import mean

device = "cuda" if torch.cuda.is_available() else "cpu"

class BertFineTune(nn.Module):
    def __init__(self, num_labels=3, vocab_size=None):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english", # using base bert model without casing (same as we have tokenized)
        )
        self.ff = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )
        self.dropout = nn.Dropout(0.1)

        for param in self.model.parameters():
            param.requires_grad = False
                  
    def forward(self, x, mask, y=None):
        outputs = self.model(
            input_ids = x,
            attention_mask = mask,
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.ff(pooled_output)

        if y is not None:
            loss = F.cross_entropy(logits, y)
            return logits, loss
        return logits

    def train_model(self, train_dataset, val_dataset, optimizer, batch_size=32):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        self.train()
        self.model.train()

        for batch in tqdm(train_dataloader, unit="batch", total=len(train_dataloader)):
            x, y, mask = batch[0], batch[1], batch[2]
            x = x.to(device, dtype = torch.long)
            y = y.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)
            
            optimizer.zero_grad()
            logits, loss = self(x, mask, y=y)
            loss.backward()
            optimizer.step()
            
        est_train_loss_loader = DataLoader(Subset(train_dataset, range(len(val_dataset))), batch_size=batch_size, shuffle=True)
        train_loss, train_accuracy = self.estimate_loss_and_accuracy(est_train_loss_loader)
        val_loss, val_accuracy = self.estimate_loss_and_accuracy(val_dataloader)

        return train_loss, val_loss, train_accuracy, val_accuracy

    @torch.no_grad()
    def estimate_loss_and_accuracy(self, dataloader):
        self.eval()
        self.model.eval()
        losses, accs = [], []
        for (x, y, mask) in dataloader:
            x = x.to(device, dtype = torch.long)
            y = y.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)
            logits, loss = self(x, mask, y=y)
            losses.append(loss.item())
            accs.append(accuracy_score(y.cpu().numpy(), torch.argmax(logits, dim=-1).cpu().numpy()))
        return round(mean(losses), 4), round(mean(accs), 4)
    
    def setFineTune(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def eval_model(self, test_dataset, batch_size=32):
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        metrics, loss = self.get_metrics(test_dataloader)
        return metrics, loss

    @torch.no_grad()
    def get_metrics(self, dataloader):
        self.eval()
        self.model.eval()
        losses, ys, preds = [], [], []
        for (x, y, mask) in dataloader:
            x = x.to(device, dtype = torch.long)
            y = y.to(device, dtype = torch.long)
            mask = mask.to(device, dtype = torch.long)
            ys += y.tolist()
            logits, loss = self(x, mask, y=y)
            preds += torch.argmax(logits, dim=-1).tolist()
            losses.append(loss.item())

        ys = np.array(ys)
        preds = np.array(preds)
        target_names = ['dissapointed', 'happy', 'angry']
        metrics = classification_report(ys, preds, target_names=target_names, digits=3)
        
        return metrics, mean(losses)
    
    def get_dataset(self, data, labels, tokenizer, add_special_tokens = False, remove_emojis=False, tag=""):
        data_encoded, attention_masks = tokenizer.encode_many(data, add_special_tokens=add_special_tokens,
                                                              remove_emojis=remove_emojis, tag=tag)
        return TwitterDatasetForBert(data_encoded, labels, attention_masks)
