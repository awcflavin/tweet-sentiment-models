from statistics import mean
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
import math
from torch.nn import functional as F
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from utils.datasets import TwitterDataset, TwitterGenTrainDataset
from utils.utils import pad_tweets

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# attention head - can be masked or not
class Head(nn.Module):
    def __init__(self, head_size, emb_dim, block_size=None):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.block_size = block_size
        if self.block_size:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        weights = q@k.transpose(-2, -1) * (self.head_size**-0.5) # scaling factor to keep variance constant

        if self.block_size:
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # [:T, :T] allows handling of variable sized context lengths

        weights = F.softmax(weights, dim=-1)
        out = weights @ v
        return out

# multiple attention heads concatenated
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, emb_dim, block_size=None, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, emb_dim, block_size=block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # cat all head outputs
        out = self.proj(out) # linear projection out
        out = self.dropout(out)
        return out

# feed forward network
class FFN(nn.Module):
    def __init__(self, features, dropout=0.1):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(features, 4*features),
                                    nn.GELU(),
                                    nn.Linear(4*features, features),
                                    nn.Dropout(dropout),)
    
    def forward(self, x):
        return self.layer(x)

# transformer block
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_head, block_size=None, dropout=0.1):
        super().__init__()
        head_size = emb_dim // n_head
        self.heads = MultiHeadAttention(n_head, head_size, emb_dim, block_size=block_size, dropout=dropout)
        self.ffn = FFN(emb_dim, dropout=dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    # skip connections and layer norm applied before each sub-block as in gpt-2
    def forward(self, x):
        x = x + self.heads(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        x = self.dropout(x)
        return x

# encoder only classifier model
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=32, block_size=128, num_classes=3, heads=1, blocks=1, dropout=0.1, use_learned_positional_embeddings=True):
        super().__init__()
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.use_learned_positional_embeddings = use_learned_positional_embeddings
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(emb_dim, heads, dropout=dropout) for _ in range(blocks)])
        self.norm = nn.LayerNorm(emb_dim)
        self.linear_out = nn.Linear(emb_dim*block_size, num_classes)
        if self.use_learned_positional_embeddings:
            self.pos_embeddings = nn.Embedding(block_size, emb_dim)
        else:
            self.pos_embeddings = self.getPosEmbeddings()
        self.dropout = nn.Dropout(dropout)

    def getPosEmbeddings(self):
        PE = torch.zeros(self.block_size, self.emb_dim).to(device, dtype=torch.float32)
        for pos in range(self.block_size):
            for i in range(self.emb_dim):
                if i % 2 == 0:
                    PE[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.emb_dim)))
                else:
                    PE[pos, i] = math.cos(pos / (10000 ** ((2 * i)/self.emb_dim)))
        return PE
    
    def forward(self, x, y=None):
        B, T = x.shape
        
        tok_embeddings = self.token_embedding(x)
        if self.use_learned_positional_embeddings:
            pos_embeddings = self.pos_embeddings(torch.arange(T).to(device, dtype=torch.long))
        else:
            pos_embeddings = self.pos_embeddings
        
        x = tok_embeddings + pos_embeddings
        
        x = self.dropout(x)
        x = self.blocks(x)
        
        x = x.view(B, -1)
        logits = self.linear_out(x) # (Batch, num_classes)

        if y is not None:
            loss = F.cross_entropy(logits, y)
            return logits, loss
        return logits

    def predict(self, tweet):
        logits = self(tweet)
        probs = F.softmax(logits, dim=-1)
        _, idx = torch.max(probs, dim=-1)
        return idx
    
    def train_model(self, train_dataset, val_dataset, optimizer, batch_size=32):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
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
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
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
            preds += torch.argmax(logits, dim=1).tolist()
            losses.append(loss.item())

        ys = np.array(ys)
        preds = np.array(preds)
        target_names = ['dissapointed', 'happy', 'angry']
        metrics = classification_report(ys, preds, target_names=target_names, digits=3)
        
        return metrics, mean(losses)
    
    def get_dataset(self, data, labels, tokenizer, add_special_tokens=False, remove_emojis=False, tag=""):
        data_encoded = tokenizer.encode_many(data, add_special_tokens, remove_emojis=remove_emojis, tag=tag)
        pad_token = tokenizer.get_pad_token()
        data_padded_encoded = pad_tweets(data_encoded, self.block_size, pad_token)

        return TwitterDataset(data_padded_encoded, labels)

# decoder only generator model
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=384, block_size=16, heads=6, blocks=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.emotion_emd_dim = emb_dim
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.emotion_embedding = nn.Embedding(3, self.emotion_emd_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(emb_dim, n_head=heads, block_size=block_size) for _ in range(blocks)])
        self.norm = nn.LayerNorm(emb_dim)
        self.linear_out = nn.Linear(emb_dim, vocab_size)
        self.positional_embedding = nn.Embedding(block_size, emb_dim)
    
    def forward(self, x, emotions, y=None):
        B, T = x.shape

        tok_embeddings = self.token_embedding(x)
        pos_embeddings = self.positional_embedding(torch.arange(T).to(device, dtype=torch.long))
        x = tok_embeddings + pos_embeddings

        e_x = emotions.unsqueeze(1).repeat(1, x.shape[1])
        e_x = self.emotion_embedding(e_x)

        x = x + e_x
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.norm(x)
        logits = self.linear_out(x)

        if y is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(logits, y)
        return logits, loss

    def generate(self, emotion, num_tokens=20, begin_token=256, end_token=257):
        ctx = [begin_token]
        ctx = torch.tensor([ctx], dtype=torch.long, device=device)
        emotion = torch.tensor([emotion], dtype=torch.long, device=device)
        tweet = []
        for _ in range(num_tokens):
            ctx = ctx[:, -self.block_size:]
            logits, loss = self(ctx, emotion)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            #idx_next = torch.argmax(probs, dim=-1)
            ctx = torch.cat((ctx, idx_next), dim=1)
            if idx_next.item() == end_token:
                break
            tweet.append(idx_next.item())
    
        return tweet
    
    def train_model(self, train_dataset, val_dataset, optimizer, batch_size=32):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        self.train()
        for step, (x, y, emotion) in tqdm(enumerate(train_dataloader), unit="batch", total=len(train_dataloader)):

            optimizer.zero_grad()
            logits, loss = self(x, emotion, y=y)

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
        for (x, y, emotion) in dataloader:
            logits, loss = self(x, emotion, y=y)
            losses.append(loss.item())
            B, T = x.shape
            y = y.view(B*T)
            accs.append(accuracy_score(y.cpu().numpy(), torch.argmax(logits, dim=-1).cpu().numpy()))
        self.train()
        return round(mean(losses), 4), round(mean(accs), 4)
    
    def eval_model(self, test_dataset, batch_size=32):
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_loss, test_accuracy = self.get_metrics(test_dataloader)
        return test_loss, test_accuracy
    
    @torch.no_grad()
    def get_metrics(self, dataloader):
        self.eval()
        losses, ys, preds = [], [], []
        for (x, y, emotion) in dataloader:
            x = x.to(device, dtype = torch.long)
            y = y.to(device, dtype = torch.long)
            logits, loss = self(x, emotion, y=y)
            B, T = x.shape
            y = y.view(B*T)
            ys += y.tolist()
            preds += torch.argmax(logits, dim=-1).tolist()
            losses.append(loss.item())

        ys = np.array(ys)
        preds = np.array(preds)

        acc = accuracy_score(ys, preds)
        
        return acc, mean(losses)
    
    def get_dataset(self, data, labels, tokenizer, add_special_tokens=True, remove_emojis=False, tag=""):
        data_encoded = tokenizer.encode_many(data, add_special_tokens, remove_emojis=remove_emojis, tag=tag)

        return TwitterGenTrainDataset(data_encoded, self.block_size, labels)